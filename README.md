<div align="center">

   <img width="400" alt="Demo usage" src="https://github.com/saharmor/voice-lab/blob/main/logo.png">

   [**Background (What, Why, Solution overview)**](https://github.com/saharmor/voice-lab?tab=readme-ov-file#background) | [**Installation**](https://github.com/saharmor/voice-lab?tab=readme-ov-file#installation) | [**Usage**](https://github.com/saharmor/voice-lab?tab=readme-ov-file#usage)

A comprehensive testing and evaluation framework for voice agents across language models, prompts, and agent personas.

<img width="800" alt="Demo usage" src="https://github.com/saharmor/voice-lab/blob/main/usage_demo.gif">

</div>

# Background
### What
Voice Lab streamlines the process of evaluating and iterating on LLM-powered agents. Whether you're looking to optimize costs by switching to a smaller model, test newly-released models, or fine-tune prompts for better performance, Voice Lab provides the tools you need to make data-driven decisions with confidence.

_While optimized for voice agents, Voice Lab is valuable for any LLM-powered agent evaluation needs._

### Why
Building and maintaining voice agents often involves:
* Manually reviewing hundreds of call logs
* Refining prompts without clear metrics
* Risking a performance hit when switching to new language models
* Limited ability to test edge cases systematically

### Solution & Use Cases
Voice Lab enables you to tackle common challenges in voice agent development:

#### Metrics & Analysis

* Define your custom metrics in JSON format and use LLM-as-a-Judge to score those metrics
* Track performance metrics across different configurations
* Monitor and intelligently choose the most cost-effective model

#### Model Migration & Cost Optimization
* Confidently switch between models (e.g., Claude Sonnet to GPT-4, or GPT-4 to GPT-4 Mini)
* Evaluate smaller, more efficient models for better cost-latency balance
* Generate comprehensive comparison tables across different models

#### Prompt & Performance Testing

* Test multiple prompt variations systematically
* Simulate and verify performance across diverse user types and interaction styles

# Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/saharmor/voice-lab.git
   cd voice-lab
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set up your environment variables by creating a .env file in the project root directory and adding the following environment variables:
    ```
    OPENAI_API_KEY=your_openai_api_key
    ```

# Usage
## Basic
For now, this library only supports the text part of a voice agent, i.e. testing the underlying language model and prompt. The the example_test.py to execute the pre-defined test:
```
python llm_testing/example_test.py
```

For more advanced configuration, you can use the [Voice Lab Configuration Editor](https://saharmor.me/voice-lab-ui/) to generate the json config files.

## Adding New Test Scenarios
You can generate test scenarios using the [Voice Lab Configuration Editor](https://saharmor.me/voice-lab-ui/) or edit `test_details.json`:

1. Open the `test_details.json` file located in the `llm_testing` directory.

2. Add a new entry for the scenario. Here’s a template you can use:
   ```json
   "chill pharmacy clerk": {
       "system_prompt": "You are a friendly pharmacy clerk assisting customers with their medication needs. Make sure to provide clear information and answer any questions.",
       "initial_message": "Hello! How can I assist you today?",
       "tool_calls": [
           {
               "type": "function",
               "function": {
                   "name": "end_conversation",
                   "description": "Call ONLY when conversation reaches clear end state by both sides exchanging farewell messages or one side explicitly stating they want to end the conversation.",
                   "strict": true,
                   "parameters": {
                       "type": "object",
                       "properties": {
                           "reason": {
                               "type": "string",
                               "description": "The specific reason why the conversation must end.",
                               "enum": [
                                   "explicit_termination_request",
                                   "service_not_available",
                                   "customer_declined_service"
                               ]
                           },
                           "who_ended_conversation": {
                               "type": "string",
                               "enum": ["agent", "callee"]
                           },
                           "termination_evidence": {
                               "type": "object",
                               "properties": {
                                   "final_messages": {
                                       "type": "array",
                                       "items": {
                                           "type": "string"
                                       }
                                   },
                                   "termination_type": {
                                       "type": "string",
                                       "enum": ["successful_completion", "early_termination"]
                                   }
                               },
                               "required": ["final_messages", "termination_type"]
                           }
                       },
                       "required": ["reason", "who_ended_conversation", "termination_evidence"]
                   }
               }
           }
       ],
       "success_criteria": {
           "required_confirmations": ["medication_info", "price"]
       },
       "persona": {
           "name": "Chill Clerk",
           "initial_message": "Hi there! What can I help you with today?",
           "description": "A relaxed pharmacy clerk who enjoys helping customers.",
           "role": "pharmacy_clerk",
           "traits": [
               "friendly",
               "patient",
               "helpful"
           ],
           "mood": "CHILL",
           "response_style": "CASUAL"
       }
   }
   ```

## Standalone eval agent
Coming soon

# Contribution ideas
- [x] Support providing agents with additional context via json, e.g. credit card details, price range, etc.
- [x] Dynamic metrics for json (e.g. `metrics.json`)
- [ ] Voice analysis (interruptions, pauses, etc.)
- [ ] Support more language models via [LiteLLM]([url](https://github.com/BerriAI/litellm))
- [ ] Integrate [Tencent's 1B Personas](https://huggingface.co/datasets/proj-persona/PersonaHub) for more detailed and complex scenarios
- [ ] Use Microsoft's new [TinyTroupe](https://github.com/microsoft/TinyTroupe) for more extensive simulations
- [ ] Integrate [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio) for audio analysis
- [ ] Batch processing for lower cost (50% off)
- [ ] Suggest fine-tuned models for better adherence/style/etc. evaluation (e.g., defining what is concise vs. length)
- [ ] Improve test framework
  - [ ] Create a DB of agents and personas, each with additional context (e.g. address) according to scenarios (e.g. airline, commerce)
  - [ ] Add parallel test execution
  - [ ] Add detailed test reporting
  - [ ] Add conversation replay capability
- [ ] Generated test report
  - [ ] Add the eval_metrics.json and test_scenarios that were used for the test run

# Attribution
If you use this project, please provide attribution by linking back to this repository: [https://github.com/saharmor/voice-lab](https://github.com/saharmor/voice-lab).
