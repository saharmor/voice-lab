from datetime import datetime
from itertools import product
import os
import json
import webbrowser
from core.agent_config import AgentTaskConfig
from core.personas import CalleePersona
from core.data_types import EntitySpeaking, TestResult, TestedComponent, TestedComponentType
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

def get_metric_success_indicator(metric):
    """Returns success indicator emoji based on metric evaluation"""
    if metric.eval_output_type == "success_flag":
        return "✅" if metric.eval_output.lower() == "true" else "❌"
    else:  # range_score
        return "✅" if int(metric.eval_output) > metric.eval_output_success_threshold else "❌"
    
    
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


def generate_test_results_report(tests_run_result: TestResult):
    # Helper function to generate a consistent color based on test name
    def get_color_for_test(test_name):
        # Convert test name to a number using sum of character codes
        hash_val = sum(ord(c) for c in test_name)
        # List of pleasant colors for borders
        colors = [
            '#818cf8',  # Indigo
            '#34d399',  # Emerald
            '#f472b6',  # Pink
            '#60a5fa',  # Blue
            '#a78bfa',  # Purple
            '#4ade80',  # Green
            '#f97316',  # Orange
            '#fbbf24',  # Amber
            '#dc2626',  # Red
            '#2563eb',  # Blue
        ]
        return colors[hash_val % len(colors)]

    # Get base test names (without variations)
    base_test_names = {name.split('_variation_')[0] for name in tests_run_result.keys()}
    
    # Generate color mapping
    color_map = {name: get_color_for_test(name) for name in base_test_names}
    
    # Start of HTML with styles
    html = """<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      font-family: system-ui, -apple-system, sans-serif;
      padding: 20px;
      line-height: 1.5;
    }
    
    .table-container {
      overflow-x: auto;
      max-width: 100%;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      border-radius: 8px;
    }
    
    table {
      border-collapse: collapse;
      width: 100%;
      background: white;
    }
    
    th {
      background: #f3f4f6;
      padding: 12px 16px;
      text-align: left;
      font-weight: 600;
      color: #374151;
      border-bottom: 2px solid #e5e7eb;
    }
    
    td {
      padding: 12px 16px;
      border-bottom: 1px solid #e5e7eb;
      color: #4b5563;
    }
    
    tr:hover {
      background: #f9fafb;
    }
    
    .result {
      font-weight: 600;
    }
    
    .success {
      color: #059669;
    }
    
    .failure {
      color: #dc2626; 
    }
    
    .reasoning {
      font-size: 0.875rem;
      color: #6b7280;
    }

    .test-group {
      border-left: 3px solid transparent;
    }

    .test-name {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .llm-badge {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 500;
      background: #f3f4f6;
      color: #374151;
    }

    .llm-column {
      min-width: 120px;
    }

    .conversation-btn {
      background-color: #6366f1;
      color: white;
      border: none;
      padding: 6px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.875rem;
      transition: background-color 0.2s;
    }

    .conversation-btn:hover {
      background-color: #4f46e5;
    }

    .modal {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0, 0, 0, 0.5);
      z-index: 1000;
    }

    .modal-content {
      position: relative;
      background-color: white;
      margin: 2% auto;
      padding: 20px;
      width: 80%;
      max-width: 800px;
      max-height: 90vh;
      overflow-y: auto;
      border-radius: 8px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .close-btn {
      position: absolute;
      right: 20px;
      top: 20px;
      font-size: 24px;
      cursor: pointer;
      color: #6b7280;
    }

    .close-btn:hover {
      color: #374151;
    }

    .conversation-container {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .message {
      padding: 12px;
      border-radius: 6px;
      max-width: 80%;
    }

    .human-message {
      background-color: #f3f4f6;
      align-self: flex-end;
    }

    .assistant-message {
      background-color: #e0e7ff;
      align-self: flex-start;
    }
"""
    
    # Add dynamic color styles for each base test
    for base_name, color in color_map.items():
        # Convert base name to a valid CSS class name
        css_class_name = base_name.lower().replace(' ', '-')
        html += f"""    .test-group-{css_class_name} {{
      border-left-color: {color};
    }}
"""
    
    html += """  </style>"""
    
    html += """
    <script>
    function showConversation(testName) {
        const modal = document.getElementById(testName + '-modal');
        modal.style.display = 'block';
    }

    function closeModal(testName) {
        const modal = document.getElementById(testName + '-modal');
        modal.style.display = 'none';
    }

    // Close modal when clicking outside
    window.onclick = function(event) {
        if (event.target.classList.contains('modal')) {
            event.target.style.display = 'none';
        }
    }
    </script>
    
</head>
<body>
  <div class="table-container">
    <table>
      <thead>
        <tr>
          <th>Test Name</th>
          <th>Conversation</th>
          <th class="llm-column">LLM</th>
"""

    # Get all unique metric names
    metric_names = set()
    for test_result in tests_run_result.values():
        metric_names.update(m.name for m in test_result['result'].evaluation_result.evaluation_results)

    # Add metric column headers in alphabetical order
    for metric in sorted(metric_names):
        html += f"\n          <th>{metric}</th>"

    html += """
        </tr>
      </thead>
      <tbody>"""

    # Sort test results by base name to group variations together
    sorted_tests = sorted(tests_run_result.items(), 
                         key=lambda x: (x[0].split('_variation_')[0], x[0]))


    modals_html = ""  # Store all modals HTML
    for test_name, tests_run_result in sorted_tests:
        # Get base test name for styling
        base_name = test_name.split('_variation_')[0]
        css_class_name = base_name.lower().replace(' ', '-')
        
        # Generate the row with appropriate test group class
        html += f"""
        <tr class="test-group test-group-{css_class_name}">
          <td>
            <div class="test-name">{test_name}</div>
          </td>
          <td>
            <button class="conversation-btn" onclick="showConversation('{test_name.replace("'", "\\'").replace('"', '\\"')}')">
              View Conversation
            </button>
          </td>
          <td class="llm-column">
            <span class="llm-badge">{tests_run_result["tested_component"][0]}</span>
          </td>"""

        # Create a dict for quick metric lookup
        metrics_dict = {m.name: m for m in tests_run_result["result"].evaluation_result.evaluation_results}

        # Add data for each metric
        for metric_name in metric_names:
            if metric_name in metrics_dict:
                metric = metrics_dict[metric_name]
                symbol = get_metric_success_indicator(metric)
                success = "Pass" if symbol == "✅" else "Fail"
                success_class = "success" if success == "Pass" else "failure"
                score = f" ({metric.eval_output})" if metric.eval_output_type == "range_score" else ""
                
                html += f"""
          <td>
            <div class="result {success_class}">{symbol} {success}{score}</div>
            <div class="reasoning">{metric.reasoning}</div>
          </td>"""
            else:
                html += """
          <td>N/A</td>"""

        html += """
        </tr>"""

        # Create modal for this test
        modals_html += f"""
        <div id="{test_name.replace('"', '&quot;')}-modal" class="modal">
          <div class="modal-content">
            <span class="close-btn" onclick="closeModal('{test_name}')">&times;</span>
            <h2>Conversation History - {test_name}</h2>
            <div class="conversation-container">"""
        
        # Add conversation messages for this test
        for message in tests_run_result["result"].conversation_history:
            role = ' '.join(word.capitalize() for word in message["speaker"].replace('_', ' ').split())
            content = message["text"]
            message_class = "human-message" if role == EntitySpeaking.CALLEE else "assistant-message"
            modals_html += f"""
              <div class="message {message_class}">
                <strong>{role}</strong><br>
                {content}
              </div>"""
            
        modals_html += """
            </div>
          </div>
        </div>"""

    # Close the table and add all modals
    html += """
      </tbody>
    </table>
  </div>"""
    
    # Add all modals after the table
    html += modals_html

    # Close the HTML document
    html += """
</body>
</html>"""

    # Save results and open the file in the browser for further analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    with open(f"{results_dir}/test_run_{timestamp}.html", "w") as f:
        f.write(html)
    
    html_path = f"file://{os.path.abspath(f'{results_dir}/test_run_{timestamp}.html')}"
    webbrowser.get('chrome').open(html_path, new=2)
