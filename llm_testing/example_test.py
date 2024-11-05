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
    # evaluator_llm = OpenAIProvider(api_key, "gpt-4o-mini")
    evaluator_llm = OpenAIProvider(api_key, "gpt-4o")
    # evaluator_llm = OpenAIProvider(api_key, "o1-preview")

    # Load test details from a JSON file
    with open('llm_testing/test_details.json', 'r') as file:  # Adjust the path as necessary
        test_details = json.load(file)

    for test_name, test_data in test_details.items():
        print(f"\n=== Running Test: {test_name} ===")

        agent_config = test_data["agent"]
        goal = AgentTaskConfig(
            system_prompt=agent_config["system_prompt"],
            initial_message=agent_config["initial_message"],
            tool_calls=agent_config["tool_calls"],
            success_criteria=agent_config["success_criteria"],
            additional_context=agent_config["additional_context"]
        )
        
        persona = CalleePersona(**test_data["persona"])

        evaluator = LLMConversationEvaluator(evaluator_llm)
        runner = GoalBasedTestRunner(agent_llm, evaluator)

        eval_response = runner.run_conversation_test(goal, persona, max_turns=50)

        print("\n=== Evaluation report ===")
        print(f"Summary: {eval_response.summary}")
        for metric in eval_response.evaluation_results:
            # add a ✅ emoji if the metric is successful based on the eval_output_type
            success_indicator = ""
            if metric.eval_output_type == "success_flag":
                success_indicator = "✅" if metric.eval_output == "True" else "❌"
            else: # range_score
                success_indicator = "✅" if int(metric.eval_output) > metric.range_score_success_threshold else "❌"

            print(f"--> {success_indicator} Metric: {metric.name}, Output score: {metric.eval_output}\nReasoning: {metric.reasoning}\nEvidence: {metric.evidence}\n")
            
        if print_conversation:
            print("\nConversation History:")
            for turn in runner.conversation_history:
                print(f"{turn['speaker'].upper()}: {turn['text']}")

    return eval_response

if __name__ == "__main__":
    load_dotenv()
    evaluation = run_tests()