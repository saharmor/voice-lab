import json
import os

from core.providers.openai import OpenAIProvider


url = "https://substack.com/support"

#TODO remove once connected to browser automation
agent_system_prompt = "You are a customer service agent for Substack. You are doing your best to resolve users' issues as smoothly as possible."

persona = "My name is Sahar. I run a substack called AI Tidbits. I am an angry person who is impatient when it comes to customer service."
system_prompt = f'''You are my virtual assistant who contacts customer support on my behalf. About me: {persona}
Generate your next response for the following conversation so I can send it to the customer support agent.
'''
scenario = "Remove a paid subscription I suspect is fraudulent"

def read_mock_web_conv(user_turns=3):
    with open("mock_web_conv.json", "r") as f:
        mock_web_conv = json.load(f)
    
    # return the list of message until the user_turns message
    messages = []
    for msg in mock_web_conv["messages"]:
        if msg["role"] == "user":
            user_turns -= 1
        if user_turns == 0:
            break
        messages.append(msg)
    
    return messages



def print_conversation_history(conversation_history):
    for msg in conversation_history:
        print(f"{msg['role']}: {msg['content']}")

def mock_read_agent_response(messages_thus_far):
    issue_resolved_tool = {
    "type": "function",
    "function": {
        "name": "user_issue_resolved",
        "description": "Determines if a user's issue has been fully resolved. This should ONLY be called when ALL of the following conditions are met:\n1. All requested actions have been completed (not just promised)\n2. The user has explicitly confirmed their satisfaction or indicated no further help is needed\n3. There are no pending updates or actions the user is waiting for",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_resolved": {
                    "type": "boolean",
                    "description": "Whether ALL requested actions are complete AND user has confirmed satisfaction"
                },
                "satisfaction_score": {
                    "type": "number",
                    "description": "User satisfaction score (1-5), based on explicit positive confirmation from user. Default to null if unclear.",
                    "minimum": 1,
                    "maximum": 5
                },
                "confirmation_type": {
                    "type": "string",
                    "enum": ["explicit", "implicit", "none"],
                    "description": "How the user confirmed resolution: 'explicit' (clear confirmation), 'implicit' (positive but indirect), or 'none' (pending)"
                }
            },
                "required": ["satisfaction_score", "issue_resolved", "confirmation_type"]
            }
        }
    }

    return agent_llm.plain_call(agent_system_prompt,
                                    messages_thus_far,
                                    [issue_resolved_tool])

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

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

conversation_history = read_mock_web_conv()
print_conversation_history(conversation_history)

agent_llm = OpenAIProvider(api_key, "gpt-4o")

issue_resolved = False
while not issue_resolved: 
    user_response = agent_llm.plain_call(system_prompt, convert_conv_history_to_openai_format(conversation_history, "user"))
    print("user: ", user_response.response_content)
    conversation_history.append({"role": "user", "content": user_response.response_content})
    
    agent_response = mock_read_agent_response(convert_conv_history_to_openai_format(conversation_history, "agent"))
    print("assistant: ", agent_response.response_content)
    conversation_history.append({"role": "agent", "content": agent_response.response_content})

    if agent_response.tools_called:
        tool_call = agent_response.tools_called[0]
        if tool_call.function.name == "user_issue_resolved":
            func_args = json.loads(tool_call.function.arguments)
            issue_resolved = func_args["issue_resolved"]
            if issue_resolved:
                issue_resolved = True
                satisfaction_score = func_args["satisfaction_score"]
                print(f"User's issue has been resolved with a satisfaction score of {satisfaction_score}")




