from itertools import product
import os
import json
from core.agent_config import AgentTaskConfig
from core.personas import CalleePersona
from test_runner import GoalBasedTestRunner
from core.evaluator import LLMConversationEvaluator
from providers.openai import OpenAIProvider


def generate_test_combinations(test_data):
    # Sort underlying LLMs first, system prompts later
    sorted_llms = sorted(test_data["tested_components"]["underlying_llms"])
    sorted_prompts = sorted(test_data["tested_components"]["agent_system_prompts"])
    return list(product(sorted_llms, sorted_prompts))


def run_tests(print_conversation: bool = False):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")


    # evaluator_llm = OpenAIProvider(api_key, "gpt-4o-mini")
    evaluator_llm = OpenAIProvider(api_key, "gpt-4o")
    # evaluator_llm = OpenAIProvider(api_key, "o1-preview")

    # Load test details from a JSON file
    with open('llm_testing/config/test_scenarios.json', 'r') as file:  # Adjust the path as necessary
        test_scenarios = json.load(file)

    tests_results = {}
    test_number = 1
    for test_name, test_data in test_scenarios.items():
        print(f"\n=== Running Test: #{test_number} - {test_name} ===")

        agent_config = test_data["agent"]
        # create a matrix of all tested components and possible combinations
        tested_components = generate_test_combinations(test_data)   

        for tested_component_variation in tested_components:
            agent_llm = OpenAIProvider(api_key, tested_component_variation[0])
            agent_prompt = tested_component_variation[1]
            print(f"Tested component: [{tested_component_variation[0]}] + [{tested_component_variation[1]}]")

            agent_task_config = AgentTaskConfig(
                system_prompt=agent_prompt,
                initial_message=agent_config["initial_message"],
                tool_calls=agent_config["tool_calls"],
                success_criteria=agent_config["success_criteria"],
                additional_context=agent_config["additional_context"]
            )
        
            persona = CalleePersona(**test_data["persona"])

            evaluator = LLMConversationEvaluator(evaluator_llm)
            runner = GoalBasedTestRunner(agent_llm, evaluator)

            eval_response = runner.run_conversation_test(agent_task_config, persona, max_turns=50)
            tests_results[test_name] = {
                "tested_component": tested_component_variation,
                "result": eval_response
            }

            print("\n=== Evaluation report ===")
            print(f"Summary: {eval_response.summary}\n")
            for metric in eval_response.evaluation_results:
                    success_indicator = ""
                    if metric.eval_output_type == "success_flag":
                        success_indicator = "✅" if metric.eval_output == "True" else "❌"
                    else: # range_score
                        success_indicator = "✅" if int(metric.eval_output) > metric.eval_output_success_threshold else "❌"

                    print(f"--> {success_indicator} Metric: [{metric.name}], Output score: [{metric.eval_output}]\nReasoning: {metric.reasoning}\nEvidence: {metric.evidence}\n")
            
            if print_conversation:
                print("\nConversation History:")
                for turn in runner.conversation_history:
                    print(f"{turn['speaker'].upper()}: {turn['text']}")
            
            print("-" * 100)
            break # TODO: remove this

    print(f"\n\n=== All tests completed: {len(tests_results)} ===")
    return tests_results