import asyncio
from datetime import datetime
import json
import os
import tempfile
import time
from typing import Optional

from pydantic import BaseModel
from pyppeteer import launch

from core.data_types import TestResult
from core.evaluator import LLMConversationEvaluator
from core.personas import CalleePersona, Mood
from core.providers.openai import OpenAIProvider
from core.utils.generate_report import generate_test_results_report

CHATBOT_REPLY_TIMEOUT_SEC = 60
FAQS_FOLDER = "faqs"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

agent_llm = OpenAIProvider(api_key, "gpt-4o-mini")

issue_resolved_tool = {
    "type": "function",
    "function": {
        "name": "user_issue_resolved",
        "description": """Call this function when any resolution pattern is detected:

1. EXPLICIT resolution:
- User states issue is resolved
- User says "thank you" with completion indication
- User confirms understanding of next steps
- User agrees to form submission or specialist help
- User says "yes" to form/specialist assistance

2. IMPLICIT resolution:
- User shows satisfaction AND confirms next steps
- User acknowledges info and states next action
- User indicates no further needs
- Form submission/specialist stage reached
- Contact form presented without objection

3. NOT resolved if:
- User still asking questions
- User seems confused
- Info incomplete
- Actions not acknowledged
- User declines form/specialist help""",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_resolved": {
                    "type": "boolean",
                    "description": "True if user satisfied AND acknowledged next steps OR agreed to form/specialist"
                },
                "reply_msg": {
                    "type": "string", 
                    "description": "What user should say next if not resolved"
                },
                "confirmation_type": {
                    "type": "string",
                    "enum": ["explicit", "implicit", "none"],
                    "description": "explicit: Clear agreement/completion, implicit: Positive response/form stage, none: Pending"
                }
            },
            "required": ["issue_resolved", "confirmation_type", "reply_msg"]
        }
    }
}

def read_mock_web_conv(scenario, user_turns=3):
    messages = []
    for msg in scenario["mock_messages"]:
        if msg["role"] == "user":
            user_turns -= 1
        if user_turns == 0:
            break
        messages.append(msg)

    return messages


def read_test_scenarios():
    with open("web_test_scenarios.json", "r") as f:
        test_scenarios = json.load(f)
    return test_scenarios["test_scenarios"]


def print_conversation_history(conversation_history):
    for msg in conversation_history:
        print(f"{msg['role']}: {msg['content']}\n")


def convert_conv_history_to_openai_format(conversation_history, assistant_role):
    msgs = []
    for msg in conversation_history:
        if msg["role"] == "agent":
            if assistant_role == "agent":
                msgs.append({"role": 'assistant', "content": msg["content"]})
            else:
                msgs.append({"role": 'user', "content": msg["content"]})
        elif msg["role"] == "user":
            if assistant_role == "user":
                msgs.append({"role": 'assistant', "content": msg["content"]})
            else:
                msgs.append({"role": 'user', "content": msg["content"]})
    return msgs


def eval_test_scenario(scenario, conversation_history):
    eval_llm = OpenAIProvider(api_key, "gpt-4o")
    evaluator = LLMConversationEvaluator(eval_llm, "eval_metrics.json",
                                         f"You are an objective conversational AI chatbot evaluator who evalutes customer support AI chatbots that text with customers. You will be provided a chat transcript and score it across the different provided metrics.")

    success_criteria = scenario["successful_outcome"]
    with open(os.path.join(FAQS_FOLDER, scenario["guidelines"]), 'r') as f:
        scenario_guidelines = f.read()

    conversation_history_str = ""
    for msg in conversation_history:
        conversation_history_str += f"{msg['role']}: {msg['content']}\n"



    user_persona = CalleePersona(
        name="User",
        description=scenario["user_persona"]["context"],
        role=scenario["user_persona"]["profession"],
        traits=[],  # Not provided in user_persona
        mood=Mood.IMPATIENT,  # Not provided in user_persona
        initial_message=scenario["user_persona"]["initial_message"],
        response_style=None,  # Not provided in user_persona
        additional_context={
            "chat_style": scenario["user_persona"]["chat_style"],
            "emotional_state": scenario["user_persona"]["emotional_state"]
        },
    )

    return TestResult(
        evaluation_result=evaluator.evaluate(
            conversation_history,
            None,
            user_persona,
            success_criteria,
            scenario_guidelines
        ),
        conversation_history=conversation_history
    )


class ChatStatus(BaseModel):
    status: str
    btn_name: Optional[str] = None


class ChatMessageWindow(BaseModel):
    status: str
    placeholder_txt: Optional[str] = None

class ChatSessionManager:
    def __init__(self, page, chat_input_selector, shadow_root_selector=None):
        self.page = page
        self.shadow_root_selector = shadow_root_selector

        # TODO get chat_input_element using VLMs instead of user to provide
        self.chat_input_selector = chat_input_selector

    async def check_if_support_chat_running(self) -> ChatStatus:
        # Ask LLM if this appears to be a support chat page
        prompt = """Analyze this webpage screenshot and determine if the chat is already running and ready to send a message.
Your response should be {status: <status_string>, btn_name: <button_name>}. btn_name is the name of the text of the button that needs to be clicked to start the chat if it exists. Otherwise, no need to return this key.
Respond with status running if the chat is already running.
Respond with status exists and the name of the text of the button that needs to be clicked to start the chat if it exists.
Respond with status unknown otherwise, i.e. if it is not clear whether the chat is running or not or you couldn't find a button to start the chat.
"""

        # Send screenshot and prompt to LLM for analysis
        screenshot = await self.page.screenshot({'fullPage': True})
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
            temp_file.write(screenshot)
            response = await agent_llm.analyze_image(temp_file.name, prompt, ChatStatus)
        
        return response

    async def get_shadow_root(self):
        if self.shadow_root_selector:
            shadow_root = await self.page.evaluateHandle('''(shadow_root_selector) => {
                const container = document.querySelector(shadow_root_selector);
                return container ? container.shadowRoot : null;
            }''', self.shadow_root_selector)
        else:
            shadow_root = None
        
        return shadow_root
    
    async def get_element_from_dom(self, selector):     
        if self.shadow_root_selector:
            shadow_root = await self.get_shadow_root()
            searched_element = await self.page.evaluateHandle('''(root, selector) => {
                if (!root) return null;
                return root.querySelector(selector);
            }''', shadow_root, selector)
        else:
            searched_element = await self.page.evaluateHandle('''(selector) => {
                return document.querySelector(selector);
            }''', selector)

        if searched_element:
            return searched_element
        else:
            raise ValueError(f"Element {selector} was not found in shadow DOM.")


    async def find_chatbot_input_element(self):
        # TODO haven't tested this yet
        prompt = """Analyze this webpage screenshot and find the placeholder text of the chat input element, e.g. "Ask a detailed question...".
    The input element is the one that contains the placeholder text and where the user can type their message.
    Your response should be {status: <status_string>, placeholder_txt: <placeholder_text>}. placeholder_txt is the placeholder text of the chat input element.
    Respond with status exists and the placeholder text of the chat input element. Respond with status unknown if you can't find the chat input element.
    """

        screenshot = await self.page.screenshot({'fullPage': True})
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
            temp_file.write(screenshot)
            response = await agent_llm.analyze_image(temp_file.name, prompt, ChatMessageWindow)
            chat_input_selector = f'input[placeholder="{response.placeholder_txt}"]'
            chat_input = await self.get_element_from_dom(chat_input_selector)

            if chat_input:
                return chat_input
            else:
                raise ValueError(f"Chat input element with placeholder text '{chat_input_selector}' was not found in shadow DOM.")


    async def get_chatbot_input_element(self):
        chat_input = await self.get_element_from_dom(self.chat_input_selector)
        if chat_input:
            return chat_input
        else:
            raise ValueError(f"Chat input element with selector '{self.chat_input_selector}' was not found in shadow DOM.")


    async def initiate_support_chat(self, chatbot_url):
        # Navigate to the chatbot and wait for page to load
        await self.page.goto(chatbot_url)
        await asyncio.sleep(3)

        # check if chat is already running, e.g. https://substack.com/support
        chat_status = await self.check_if_support_chat_running()

        # TODO REMOVE, just mock for development
        chat_status = ChatStatus(status="exists", btn_name="Start a chat")
        # chat_status = ChatStatus(status="running")

        if chat_status.status == "running":
            print("Chat is already running")
        elif chat_status.status == "exists":
            chat_button_xpath = f'//button[contains(normalize-space(.), "{
                chat_status.btn_name}")]'

            await self.page.waitForXPath(chat_button_xpath, timeout=10000)
            chat_buttons = await self.page.xpath(chat_button_xpath)
            if chat_buttons:
                await chat_buttons[0].click()
            else:
                raise ValueError(f"Chat button with text '{chat_status.btn_name}' was not found.")

            # wait for a few seconds to make sure the chat is running
            await asyncio.sleep(3)
        elif chat_status.status == "unknown":
            # save screenshot of the page
            screenshot = await self.page.screenshot({'fullPage': True})
            with open(f"{chatbot_url}_no_chat_button.png", "wb") as f:
                f.write(screenshot)
            raise ValueError(f"Can't find a way to start the chat. Please check the screenshot at {
                            chatbot_url}_no_chat_button.png")

        chat_input_element = await self.get_chatbot_input_element()
        return chat_input_element

    async def check_if_ongoing_conversation(self):
        chat_input_element = await self.get_chatbot_input_element()
        return chat_input_element is not None

    async def count_agent_msgs(self):
        shadow_root = await self.get_shadow_root() if self.shadow_root_selector else None
            
        return await self.page.evaluate('''(container) => {
            const root = container || document;
            if (!container && !document) return 0;
            // Look specifically for li elements with role-assistant class that have text content
            const elements = Array.from(root.querySelectorAll('li.role-assistant'))
                .filter(el => el.textContent.trim().length > 0);
            return elements.length;
        }''', shadow_root)

    async def get_conversation_history(self, get_last_msg=False):
        shadow_root = await self.get_shadow_root() if self.shadow_root_selector else None
        messages = await self.page.evaluateHandle('''function(container) {                                                                                 
             // If container is provided (shadow root case), use it as root
            // Otherwise use document as root
            const root = container || document;
            
            // First try to find messages in Sierra-style chat
            let messageContainer = root.querySelector('ol');
            if (messageContainer) {
                const allMessages = Array.from(messageContainer.querySelectorAll('li'))
                    .filter(li => li.getAttribute('aria-roledescription') === 'message');
                
                return allMessages.map(element => {
                    const isAssistant = element.classList.contains('role-assistant');
                    const textDiv = element.querySelector('.flex.flex-col.gap-3');
                    return {
                        sender: isAssistant ? 'assistant' : 'user',
                        messages: [textDiv ? textDiv.textContent.trim() : '']
                    };
                }).filter(turn => turn.messages[0]);
            }
            
            // For Decagon-style chat
            const messageContainer = root.querySelector("#chatbot-container");
            if (messageContainer) {
                // Get all message items including both user and assistant messages
                const allMessages = Array.from(messageContainer.querySelectorAll("[role=listitem], .flex.undefined"));
                
                return allMessages.map(function(element) {
                    // Check for spinner (typing indicator)
                    const spinner = element.querySelector(".spinner");
                    if (spinner) {
                        return {
                            sender: "assistant",
                            typing: true
                        };
                    }
                    
                    // Get message sender type from aria-label
                    const ariaLabel = element.getAttribute("aria-label");
                    const isAssistant = ariaLabel === "Message from assistant";
                    
                    // Get message text - get all paragraphs and join them
                    const textElements = element.querySelectorAll(".widget-chat-bubble-text span, .widget-chat-bubble-text p");
                    const messageText = Array.from(textElements)
                        .map(function(el) { return el.textContent.trim(); })
                        .join("\\n")
                        .trim();
                    
                    return {
                        sender: isAssistant ? "assistant" : "user",
                        messages: [messageText]
                    };
                }).filter(function(turn) { 
                    return (turn.messages && turn.messages[0]) || turn.typing;
                });
            }
            return [];
        }''', shadow_root)

        messages_info = await messages.jsonValue()
        if not messages_info:
            raise ValueError("No messages found in the conversation")
        
        if get_last_msg:
            return messages_info[-1]
        else:
            return messages_info

    async def wait_for_agent_to_finish_replying(self, get_last_msg=False):
        # TODO store latencies for (a) started typing (b) started sending/streaming message and (c) finished sending/streaming message
        # wait for the agent to start typing
        print(f"{datetime.now().strftime('%H:%M:%S')} - Waiting for agent to start typing")
        last_msg = await self.get_conversation_history(get_last_msg=True)
        time_since_started_typing = time.time()
        is_typing = False
        while last_msg['sender'] == 'user' and time.time() - time_since_started_typing < CHATBOT_REPLY_TIMEOUT_SEC:
            await asyncio.sleep(1)
            last_msg = await self.get_conversation_history(get_last_msg=True)
            if not is_typing and 'typing' in last_msg and last_msg['typing']:
                print(f"{datetime.now().strftime('%H:%M:%S')} - Agent started typing")
                is_typing = True

        if last_msg['sender'] == 'user':
            raise ValueError(f"Agent didn't reply after time {CHATBOT_REPLY_TIMEOUT_SEC} seconds")
        
        # wait for the agent to finish typing and sending messages (sometimes it sends multiple messages instead of one)
        # sleep for a bit to see if agent keeps on generating text
        await asyncio.sleep(1)
        curr_msg = await self.get_conversation_history(get_last_msg=True)
        is_typing = ('typing' in curr_msg and curr_msg['typing'])
        try:
            # keep waiting as long as (a) agent is typing or (b) agent has finished typing but want to ensure not more messages are coming (c) timout hasn't elapsed
            while (is_typing or (not is_typing and ('messages' in curr_msg and 'messages' not in last_msg))) or \
                (last_msg['messages'][-1] != curr_msg['messages'][-1]) and \
                  time.time() - time_since_started_typing < CHATBOT_REPLY_TIMEOUT_SEC:
                last_msg = await self.get_conversation_history(get_last_msg=True)
                await asyncio.sleep(1)
                curr_msg = await self.get_conversation_history(get_last_msg=True)
                is_typing = ('typing' in curr_msg and curr_msg['typing'])

            print(f"{datetime.now().strftime('%H:%M:%S')} - Agent finished typing\n\n")

            if last_msg['sender'] == 'user':
                raise ValueError(f"Agent didn't reply after time {CHATBOT_REPLY_TIMEOUT_SEC} seconds")

            if get_last_msg:
                return last_msg
        except Exception as e:
            print(f"Error while waiting for agent reply: {str(e)}")
            raise
    
        return None

    async def get_last_messages(self):
        last_msg = await self.wait_for_agent_to_finish_replying(get_last_msg=True)
        return last_msg


    async def send_and_measure(self, msg_input_element, message, typing_delay=0):
        if not message:
            raise ValueError("Message to submit to chatbot is empty")
        
        # Convert newlines to shift+enter equivalent to keep message as single input
        message = message.replace('\n', '\r')
        await msg_input_element.type(message, {'delay': typing_delay})
        await self.page.keyboard.press('Enter')

        # TODO start and end time should be caluclated in the self.get_last_messages() function as this is where we wait for response with added sleep duration
        start_time = time.time()

        # Wait for response to appear and chatbot to finish typing
        
        try:
            message_info = await self.get_last_messages()
        except Exception as e:
            print(f"Error getting messages: {str(e)}")
            response = "No response found"
        response = message_info['messages']
        end_time = time.time()

        return {
            'response': response if response != "No response found" else None,
            'latency': round(end_time - start_time, 2),
            'timestamp': datetime.now().isoformat()
        }



async def run_tests(tests_to_run_count=999, verbose=False):
    test_results = []
    browser = await launch(headless=False)
    try:
        for scenario in read_test_scenarios()[:tests_to_run_count]:
            if scenario["scenario_id"] == "refund_policy_question":
                continue
            
            try:
                user_persona = json.dumps(
                    {k: v for k, v in scenario["user_persona"].items() if k != "initial_message"})
                scenario_system_prompt = f"""You are my virtual assistant who contacts customer support on my behalf. About me: {user_persona}

Generate your next response for the following conversation so I can send it to the customer support agent.
"""
                conversation_history = []
                reply_latencies = []

                page = await browser.newPage()

                # TODO get chat_input_selector and shadow_root_selector automatically using VLMs and DOM parsing
                chat_session_manager = ChatSessionManager(page, scenario["chat_input_selector"], scenario["shadow_root_selector"])
                msg_input_element = await chat_session_manager.initiate_support_chat(scenario["chatbot_url"])
                result = await chat_session_manager.send_and_measure(msg_input_element,
                                                scenario["user_persona"]["initial_message"]
                                                # , typing_delay=10
                                                )

                if not result['response']:
                    raise ValueError(
                        f"Agent response not found for initial message (probably a selector issue)")

                if verbose:
                    print(f"Latency: {result['latency']:.2f} seconds")
                reply_latencies.append(result['latency'])

                conversation_history.append(
                    {"role": "user", "content": scenario["user_persona"]["initial_message"]})
                for msg in result['response']:
                    conversation_history.append(
                        {"role": "agent", "content": msg})

                # TODO add some timeout to cut the conversations short if the two AIs go round and round without resolving the issue
                while True:
                    user_response = agent_llm.plain_call(scenario_system_prompt,
                                                        convert_conv_history_to_openai_format(conversation_history, "user"),
                                                        [issue_resolved_tool]
                                                        )

                    # # TODO SAHAR REMOVE, just mock for development
                    # # Mock user response for development
                    # class MockResponse:
                    #     def __init__(self):
                    #         self.response_content = "test"
                    #         self.tools_called = []
                    # user_response = MockResponse()

                    if user_response.tools_called and user_response.tools_called[0].function.name == "user_issue_resolved":
                        arguments = json.loads(user_response.tools_called[0].function.arguments)
                        issue_resolved = arguments["issue_resolved"]
                        if issue_resolved:
                            print(f"{datetime.now().strftime('%H:%M:%S')} - User's issue resolved. Confirmation type: {arguments['confirmation_type']}")
                            break
                        else:
                            user_response.response_content = arguments["reply_msg"]
                            print(f"{datetime.now().strftime('%H:%M:%S')} - User's issue not resolved but tool was anyway called")
                        
                    if verbose:
                        print("user: ", user_response.response_content)
                    conversation_history.append(
                        {"role": "user", "content": user_response.response_content})

                    if not await chat_session_manager.check_if_ongoing_conversation():
                        print("Conversation ended by the chatbot")
                        break

                    # Send user generated text and then read the agent's response
                    result = await chat_session_manager.send_and_measure(msg_input_element, user_response.response_content
                                                                         # , typing_delay=10
                                                                         )

                    agent_response = result['response']
                    if not agent_response:
                        raise ValueError(
                            f"Agent response not found for user message (probably a selector issue)")

                    if verbose:
                        print("assistant: ", agent_response)
                    for msg in agent_response:
                        conversation_history.append(
                            {"role": "agent", "content": msg})

                    if verbose:
                        print(f"Latency: {result['latency']:.2f} seconds")
                    reply_latencies.append(result['latency'])

                # TODO turn into an object
                test_results.append(
                    (scenario, conversation_history, reply_latencies))
            except Exception as e:
                print(f"Error occurred when running test {scenario['scenario_id']}: {e}")
    finally:
        await browser.close()

    print(f"\nEvaluating {len(test_results)} scenarios")
    test_results_report = {}
    for scenario, conversation_history, reply_latencies in test_results:
        # TODO ADD LATENCY eval
        eval_response = eval_test_scenario(scenario, conversation_history)
        test_results_report[scenario["scenario_id"]] = {
            "tested_component": scenario,
            "result": eval_response,
            "avg_latency": sum(reply_latencies) / len(reply_latencies),
            "max_latency": max(reply_latencies)
        }
        print(f"Evaluation result: {eval_response.evaluation_result}")

    generate_test_results_report(test_results_report)
    return test_results_report




async def main():
    await run_tests(tests_to_run_count=2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
