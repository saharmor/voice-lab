# voice-lab
A testing and evaluation framework for voice agents.

<img width="800" alt="Generated report example" src="https://github.com/user-attachments/assets/b241961f-8ab0-4e98-885e-5492573faa8c">

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

## Adding a New Test Scenario

To add a new scenario (e.g., a "chill pharmacy clerk"), follow these steps:

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
# Contribution ideas
- [x] Support providing agents with additional context via json, e.g. credit card details, price range, etc.
- [x] Dyanmic metrics for json (e.g. `metrics.json`)
- [ ] Add more test scenarios
- [ ] Integrate [Tencent's 1M Personas](https://huggingface.co/datasets/proj-persona/PersonaHub) for more detailed and complex scenarios
- [ ] Batch processing for lower cost (50% off)
- [ ] Improve test framework
  - [ ] Create a DB of agents and personas, each with additional context (e.g. address) according to scenarios (e.g. airline, commerce)
  - [ ] Add parallel test execution
  - [ ] Add detailed test reporting
  - [ ] Add conversation replay capability
- [ ] Generated test report
  - [ ] Add the eval_metrics.json and test_scenarios that were used for the test
