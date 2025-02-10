import asyncio
from datetime import datetime
import json
import os
import time

from pyppeteer import launch

from core.data_types import TestResult
from core.evaluator import LLMConversationEvaluator
from core.personas import CalleePersona, Mood
from core.providers.openai import OpenAIProvider
from core.utils.generate_report import generate_test_results_report

CHATBOT_REPLY_TIMEOUT_SEC = 60


def read_mock_web_conv(scenario, user_turns=3):
    messages = []
    for msg in scenario["mock_messages"]:
        if msg["role"] == "user":
            user_turns -= 1
        if user_turns == 0:
            break
        messages.append(msg)

    return messages



issue_resolved_tool = {
    "type": "function",
    "function": {
        "name": "user_issue_resolved",
        "description": "Determines if a user's issue or question has been fully resolved. Call this function when ANY of these resolution patterns are detected:\n\n1. EXPLICIT resolution:\n- User clearly states the issue is resolved\n- User says 'thank you' and indicates they got what they needed\n- User confirms they understand next steps\n\n2. IMPLICIT resolution:\n- User expresses satisfaction ('helpful', 'great', etc.) AND confirms next steps\n- User acknowledges the information and states their intended action\n- User says they don't need anything else\n\n3. Do NOT consider resolved if:\n- User is still asking questions\n- User seems confused or uncertain\n- Information provided was incomplete\n- User needs to take actions but hasn't acknowledged them",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_resolved": {
                    "type": "boolean",
                    "description": "True if user expressed satisfaction AND acknowledged next steps (if any)"
                },
                "confirmation_type": {
                    "type": "string",
                    "enum": ["explicit", "implicit", "none"],
                    "description": "explicit: Clear statement of satisfaction/completion\nimplicit: Positive response + stated next steps\nnone: Still pending or unclear"
                }
            },
            "required": ["issue_resolved", "confirmation_type"]
        }
    }
}


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
    with open(scenario["guidelines"], 'r') as f:
        scenario_guidelines = f.read()

    conversation_history_str = ""
    for msg in conversation_history:
        conversation_history_str += f"{msg['role']}: {msg['content']}\n"

#     eval_prompt = f"""You are an objective conversational AI chatbot evaluator who evalutes customer support AI chatbots that text with customers. You will be provided a chat transcript and score it across the different provided metrics.
# Evaluate the following conversation according to Notion's customer support guidelines (attached below) and provide a score according to the scoring format and an explanation of your evaluation for each metric.
# success_flag is a boolean value that indicates whether the metric was achieved. range_score is a number between 0 and 10 that indicates the degree to which the metric was achieved.

# # Metrics
# {evaluator._generate_metrics_prompt()}

# # Success criteria
# {success_criteria}

# # Guidelines
# {scenario_guidelines}

# # Conversation
# {conversation_history_str}
# """

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

    
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

agent_llm = OpenAIProvider(api_key, "gpt-4o")


async def run_tests(tests_to_run_count=999, verbose=False):
    test_results = []
    browser = await launch(headless=False)
    try:
        for scenario in read_test_scenarios()[:tests_to_run_count]:
            try:
                user_persona = json.dumps(
                    {k: v for k, v in scenario["user_persona"].items() if k != "initial_message"})
                issue_resolved = False
                system_prompt = f"""You are my virtual assistant who contacts customer support on my behalf. About me: {user_persona}

        Generate your next response for the following conversation so I can send it to the customer support agent.
        """
                conversation_history = []
                reply_latencies = []

                page = await browser.newPage()

                msg_input_selector = 'textarea[placeholder="Ask a detailed question..."]'
                # Navigate to the chatbot
                await page.goto(scenario["chatbot_url"])

                # wait for the page to load
                await asyncio.sleep(5)

                # Wait for the chat interface to load
                await page.waitForSelector(msg_input_selector)

                result = await send_and_measure(page, msg_input_selector,
                                                scenario["user_persona"]["initial_message"]
                                                # , typing_delay=10
                                                )
                
                if not result['response']:
                    raise ValueError(f"Agent response not found for initial message (probably a selector issue)")
                
                if verbose:
                    print(f"Latency: {result['latency']:.2f} seconds")
                reply_latencies.append(result['latency'])

                conversation_history.append(
                    {"role": "user", "content": scenario["user_persona"]["initial_message"]})
                for msg in result['response']:
                    conversation_history.append({"role": "agent", "content": msg})

                while not issue_resolved:
                    user_response = agent_llm.plain_call(system_prompt,
                                                        convert_conv_history_to_openai_format(conversation_history, "user"),
                                                        [issue_resolved_tool]
                                                        )
                    
                    if user_response.tools_called and user_response.tools_called[0].function.name == "user_issue_resolved":
                        arguments = json.loads(user_response.tools_called[0].function.arguments)
                        issue_resolved = arguments["issue_resolved"]
                        print("User's issue resolved?", issue_resolved)    
                        break

                    if verbose:
                        print("user: ", user_response.response_content)
                    conversation_history.append({"role": "user", "content": user_response.response_content})

                    if not await check_if_ongoing_conversation(page, msg_input_selector):
                        print("Conversation ended by the chatbot")
                        break

                    # Send user generated text and then read the agent's response
                    result = await send_and_measure(page, msg_input_selector, user_response.response_content
                                                    # , typing_delay=10
                                                    )
                    
                    time.sleep(3)
                    
                    agent_response = result['response']
                    if not agent_response:
                        raise ValueError(f"Agent response not found for user message (probably a selector issue)")
                    
                    if verbose:
                        print("assistant: ", agent_response)
                    for msg in agent_response:
                        conversation_history.append({"role": "agent", "content": msg})

                    if verbose:
                        print(f"Latency: {result['latency']:.2f} seconds")
                    reply_latencies.append(result['latency'])
                
                # TODO turn into an object
                test_results.append((scenario, conversation_history, reply_latencies))
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

async def check_if_ongoing_conversation(page, msg_input_selector):
    return await page.evaluate(f'''() => {{
        const input = document.querySelector('{msg_input_selector}');
        if (input) return true;
        return false;
    }}''')


async def send_and_measure(page, msg_input_selector, message, typing_delay=0):
    # Clear input if needed
    await page.evaluate(f'''() => {{
        const input = document.querySelector('{msg_input_selector}');
        if (input) input.value = '';
    }}''')

    # Convert newlines to shift+enter equivalent to keep message as single input
    message = message.replace('\n', '\r')
    await page.type(msg_input_selector, message, {'delay': typing_delay})
    await page.keyboard.press('Enter')
    start_time = time.time()

    # Wait for response to appear and chatbot to finish typing and wait for spinner to disappear
    # TODO spinner can sometimes disappear and reappear as the agent is thinking. Wait for a few second to (a) check if there are new messages (i.e. multiple) or (b) agent is still thinking
    
    # Wait for initial spinner to appear
    await page.waitForSelector('.spinner', {'timeout': 15000})
    
    spinner_disappeared = False
    start_wait = time.time()
    
    while not spinner_disappeared:
        try:
            # Wait for spinner to disappear
            await page.waitForFunction(
                '!document.querySelector(".spinner")',
                {'timeout': CHATBOT_REPLY_TIMEOUT_SEC * 1000}
            )

            # Wait a bit to see if spinner reappears
            time.sleep(1.5)
            
            # Check if spinner is still gone
            spinner_disappeared = await page.evaluate('!document.querySelector(".spinner")')
        except Exception:
            # Spinner reappeared or timeout, continue loop
            continue
            
    if not spinner_disappeared:
        raise TimeoutError("Agent response timed out")

    end_time = time.time()

    # Get the agent's latest response
    response = await page.evaluate('''() => {
        const messages = document.querySelectorAll('.widget-chat-bubble');
        const agentMessages = [];
        let lastUserMessage = -1;
        
        // First find index of last user message
        for (let i = 0; i < messages.length; i++) {
            if (!messages[i].classList.contains('bg-slate-200')) {
                lastUserMessage = i;
            }
        }
        
        // Get agent messages after last user message
        for (let i = lastUserMessage + 1; i < messages.length; i++) {
            if (messages[i].classList.contains('bg-slate-200')) {
                const textElement = messages[i].querySelector('.widget-chat-bubble-text');
                if (textElement) {
                    agentMessages.push(textElement.innerText);
                }
            }
        }
        
        return agentMessages.length > 0 ? agentMessages : 'No response found';
    }''')


    return {
        'response': response if response != "No response found" else None,
        'latency': end_time - start_time,
        'timestamp': datetime.now().isoformat()
    }


async def main():
    await run_tests(tests_to_run_count=2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
