from typing import Dict
from ..data_types import TestResult
import os
import webbrowser
from datetime import datetime


def get_metric_success_indicator(metric):
    """Returns success indicator emoji based on metric evaluation"""
    if metric.eval_output_type == "success_flag":
        return "✅" if metric.eval_output.lower() == "true" else "❌"
    else:  # range_score
        return "✅" if int(metric.eval_output) > metric.eval_output_success_threshold else "❌"


def generate_test_results_report(tests_run_result: Dict[str, TestResult]):
    if not tests_run_result:
        return
    
    # TODO fix, pig
    test_type = "llm_testing" if "start_timestamp" in tests_run_result[list(
        tests_run_result.keys())[0]]['result'].conversation_history[0] else "web_eval"

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
    base_test_names = {name.split('_variation_')[
        0] for name in tests_run_result.keys()}

    # Generate color mapping
    color_map = {name: get_color_for_test(name) for name in base_test_names}

    css_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "html_report_style.css")
    with open(css_path, 'r') as f:
        css_content = f.read()

    # Start of HTML
    html = """<!DOCTYPE html>
<html>
<head>
  <style>
"""
    html += css_content

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
"""

    if test_type == "web_eval":
        html += f"<th class='llm-column'>Chatbot URL</th>"
        html += f"<th class='llm-column'>Agent Avg Latency</th>"
        html += f"<th class='llm-column'>Agent Max Latency</th>"
    else:
        html += f"<th class='llm-column'>LLM</th>"

    # Get all unique metric names
    metric_names = set()
    for test_result in tests_run_result.values():
        metric_names.update(
            m.name for m in test_result['result'].evaluation_result.evaluation_results)

    # Add metric column headers in alphabetical order
    metric_names = sorted(metric_names)
    for metric in metric_names:
        html += f"\n          <th>{metric}</th>"

    html += """
        </tr>
      </thead>
      <tbody>"""

    # Sort test results by base name to group variations together
    sorted_tests = sorted(tests_run_result.items(),
                          key=lambda x: (x[0].split('_variation_')[0], x[0]))

    modals_html = ""  # Store all modals HTML
    for test_name, tests_components_result in sorted_tests:
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
          """
        if test_type == "web_eval":
            html += f"""<td>
            {f'<span class="llm-badge">{tests_components_result["tested_component"]["chatbot_url"]}</span>'}
            </td>
            
            <td>
            {f'<span>{tests_components_result["avg_latency"]:.2f}</span>'}
            </td>
            
            <td>
            {f'<span>{tests_components_result["max_latency"]:.2f}</span>'}
            </td>
          """
        else:
            html += f"""
          <td class="llm-column">
            {f'<span class="llm-badge">{tests_components_result["tested_component"]
                                        [0]}</span>' if tests_components_result["tested_component"] else ''}
          </td>"""

        # Create a dict for quick metric lookup, sorted alphabetically by metric name
        metrics_dict = {
            m.name: m for m in tests_components_result["result"].evaluation_result.evaluation_results}

        # Add data for each metric
        for metric_name in metric_names:
            if metric_name in metrics_dict:
                metric = metrics_dict[metric_name]
                symbol = get_metric_success_indicator(metric)
                success = "Pass" if symbol == "✅" else "Fail"
                success_class = "success" if success == "Pass" else "failure"
                score = f" ({
                    metric.eval_output})" if metric.eval_output_type == "range_score" else ""

                html += f"""
          <td>
            <div class="result {success_class}">{symbol} {success}{score}</div>
                <div class="reasoning">{metric.reasoning.replace('\n', '<br>')}</div>
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
        # TODO content and role are web_eval's version. Should be unified to work with the llm_testing version (text and EntitySpeaking)
        for message in tests_components_result["result"].conversation_history:
            role = ' '.join(word.capitalize()
                            for word in message["role"].replace('_', ' ').split())
            content = message["content"]
            message_class = "human-message" if role == 'User' else "assistant-message"
            if test_type != "web_eval":
                start_timestamp_html = f"{
                    message['start_timestamp']:.1f}s" if 'start_timestamp' in message else ""
                end_timestamp_html = f"{
                    message['end_timestamp']:.1f}s" if 'end_timestamp' in message else ""
                if start_timestamp_html and end_timestamp_html:
                    timestamp_html = f"[{
                        start_timestamp_html} -> {end_timestamp_html}] "
            else:
                timestamp_html = ""

            modals_html += f"""
              <div class="message {message_class}">
                <strong>{timestamp_html}{role}</strong><br>
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
    results_dir = "./test_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(f"{results_dir}/test_run_{timestamp}.html", "w") as f:
        f.write(html)

    html_path = f"file://{os.path.abspath(
        f'{results_dir}/test_run_{timestamp}.html')}"
    webbrowser.get('chrome').open(html_path, new=2)
