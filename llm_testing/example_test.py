import os
from dotenv import load_dotenv
import json
from core.goals import AgentTaskConfig
from core.personas import CalleePersona
from test_runner import GoalBasedTestRunner
from core.evaluator import LLMConversationEvaluator
from providers.openai import OpenAIProvider

def run_tests(print_conversation: bool = False):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Create LLM instances for agent and evaluator
    agent_llm = OpenAIProvider(api_key, "gpt-4o-mini")
    evaluator_llm = OpenAIProvider(api_key, "gpt-4o-mini")

    # Load test details from a JSON file
    with open('llm_testing/test_details.json', 'r') as file:  # Adjust the path as necessary
        test_details = json.load(file)

    for test_name, test_data in test_details.items():
        print(f"\n=== Running Test: {test_name} ===")

        goal = AgentTaskConfig(
            system_prompt=test_data["system_prompt"],
            initial_message=test_data["initial_message"],
            tool_calls=test_data["tool_calls"],
            success_criteria=test_data["success_criteria"]
        )
        
        persona = CalleePersona(**test_data["persona"])

        evaluator = LLMConversationEvaluator(evaluator_llm)
        runner = GoalBasedTestRunner(agent_llm, evaluator)

        evaluation = runner.run_conversation_test(goal, persona, max_turns=50)

        print("\n=== Test Results ===")
        print(f"Goal Achieved: {evaluation.goal_achieved}")
        print(f"\nReasoning: {evaluation.reasoning}")
        if print_conversation:
            print("\nConversation History:")
            for turn in runner.conversation_history:
                print(f"{turn['speaker'].upper()}: {turn['text']}")

    return evaluation

if __name__ == "__main__":
    load_dotenv()
    evaluation = run_tests()