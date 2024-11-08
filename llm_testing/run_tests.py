from itertools import product
import os
import json
from core.agent_config import AgentTaskConfig
from core.personas import CalleePersona
from core.data_types import TestedComponent, TestedComponentType
from utils.generate_report import get_metric_success_indicator
from test_runner import GoalBasedTestRunner
from core.evaluator import LLMConversationEvaluator
from providers.openai import OpenAIProvider


def generate_test_combinations(test_data):
    # Sort underlying LLMs first, system prompts later
    tested_components_data = test_data["tested_components"]
    tested_components = [
        TestedComponent(
            type=TestedComponentType.UNDERLYING_LLM,
            variations=sorted(tested_components_data["underlying_llms"])
        ),
        TestedComponent(
            type=TestedComponentType.AGENT,
            variations=sorted(tested_components_data["agent_system_prompts"])
        )
    ]
    sorted_llms = tested_components[0].variations
    sorted_prompts = tested_components[1].variations

    return list(product(sorted_llms, sorted_prompts))    
    
def run_tests(tests_to_run: list[str] = [], print_verbose: bool = False):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # To choose the best LLM-as-a-Judge, review https://arxiv.org/abs/2410.12784 and https://huggingface.co/spaces/ScalerLab/JudgeBench
    evaluator_llm = OpenAIProvider(api_key, "gpt-4o-mini")
    # evaluator_llm = OpenAIProvider(api_key, "gpt-4o")
    # evaluator_llm = OpenAIProvider(api_key, "o1-preview")

    # Load test details from a JSON file
    with open('llm_testing/config/test_scenarios.json', 'r') as file:  # Adjust the path as necessary
        test_scenarios = json.load(file)

    tests_results = {}
    test_number = 1
    for test_name, test_data in test_scenarios.items():
        if tests_to_run and test_name not in tests_to_run:
            continue
        
        print(f"\n=== Running Test: #{test_number} - {test_name} ===")

        agent_config = test_data["agent"]
        # Create a matrix of all tested components and possible combinations
        tested_components = generate_test_combinations(test_data)   

        for tested_component_variation in tested_components:
            agent_llm = OpenAIProvider(api_key, tested_component_variation[0])
            agent_prompt = tested_component_variation[1]
            print(f"Tested component: [{tested_component_variation[0]}] + [{tested_component_variation[1][:50]}...]")

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

            eval_response = runner.run_conversation_test(agent_task_config, persona, max_turns=50) # TODO: remove max_turns
            tests_results[f"{test_name}_variation_{test_number}"] = {
                "tested_component": tested_component_variation,
                "result": eval_response
            }
            
            if print_verbose:
                print("\n=== Evaluation report ===")
                print(f"Summary: {eval_response.summary}\n")
                for metric in eval_response.evaluation_results:
                    success_indicator = get_metric_success_indicator(metric)
                    print(f"--> {success_indicator} Metric: [{metric.name}], Output score: [{metric.eval_output}]\nReasoning: {metric.reasoning}\nEvidence: {metric.evidence}\n")

                print("\nConversation History:")
                for turn in runner.conversation_history:
                    print(f"{turn['speaker'].upper()}: {turn['text']}")

            test_number += 1
            print(f"\n{'-' * 100}\n")

            # break # TODO: remove this, just for dev

    print(f"\n\n=== All tests completed: {test_number - 1} ===")
    return tests_results


