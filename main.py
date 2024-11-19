import os
from llm_testing.core.agent_config import AgentTaskConfig
from llm_testing.core.data_types import EntitySpeaking, MetricResult, TestResult, TestedComponent, TestedComponentType
from llm_testing.core.evaluator import LLMConversationEvaluator
from llm_testing.core.personas import CalleePersona, Mood, ResponseStyle
from llm_testing.providers.openai import OpenAIProvider
from llm_testing.utils.generate_report import generate_test_results_report
from speech_testing.data_types import Speaker
from speech_testing.run_tests import run_tests as run_speech_tests

from dotenv import load_dotenv

from speech_testing.utils import generate_mock_test_result

load_dotenv()

def suppress_output():
    import warnings
    import logging
    import os
    from tqdm import tqdm

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Suppress logging messages
    logging.getLogger().setLevel(logging.CRITICAL)

    # Suppress PyTorch Lightning version warning
    os.environ["PYTORCH_LIGHTNING_SUPPRESS"] = "1"
    os.environ["LIGHTNING_SUPPRESS_LOGGING"] = "1"

    # Redirect stdout to suppress tqdm output
    import sys
    class DummyFile(object):
        def write(self, x): pass
        def flush(self): pass

    sys.stdout = DummyFile()  # Suppress tqdm
    tqdm.monitor_interval = 0  # Disable tqdm warning


suppress_output()

# Run text-based tests
# test_result = run_llm_tests()
# generate_test_results_report(test_result)

# Run speech-based tests
tests_result = run_speech_tests(["interruption_test.wav"])
# test_result = generate_mock_test_result()


# Evaluate existing call with text evals (i.e. LLMs) so no need to simulate
api_key = os.getenv("OPENAI_API_KEY")
evaluator_llm = OpenAIProvider(api_key, "gpt-4o-mini")
evaluator = LLMConversationEvaluator(evaluator_llm)

# generate mock test_result

conversation_history = []
completed_tests = []
test_result = tests_result[0]
for call_segment in test_result.call_segments:
    conversation_history.append({
        "speaker": EntitySpeaking.CALLEE.value if call_segment.speaker == Speaker.CALLEE else EntitySpeaking.VOICE_AGENT.value,
        "text": call_segment.text
    })

evaluation_result = evaluator.evaluate(
    conversation_history,
    task_config=AgentTaskConfig(
        system_prompt="Order a pizza",
        initial_message="Hi, I'd like to order a pizza.",
        tool_calls=[],
        success_criteria={
            "pizza_ordered": "The customer has ordered a pizza."
        }
    ),
    persona=CalleePersona(
        name="Sarah",
        role="callee",
        description="A helpful and friendly pizza person",
        traits=["helpful", "friendly"],
        initial_message="Hi, I'd like to order a pizza.",
        mood=Mood.HAPPY,
        response_style=ResponseStyle.FRIENDLY,
        additional_context={})
)

evaluation_result.evaluation_results.append(
    MetricResult(name="interruptions", eval_output_type="success_flag",
                  eval_output="true" if len(test_result.interruptions) == 0 else "false",
                  eval_output_success_threshold=1,
                  reasoning=f"Had {len(test_result.interruptions)} interruptions.",
                  evidence="")
)
evaluation_result.evaluation_results.append(
    MetricResult(name="pauses", eval_output_type="success_flag",
                  eval_output="true" if len(test_result.pauses) == 0 else "false",
                  eval_output_success_threshold=1,
                  reasoning=f"Had {len(test_result.pauses)} pauses.",
                  evidence="")
)

generate_test_results_report(
    {
        "audio_mock_test": {
            "tested_component": [TestedComponent(type=TestedComponentType.UNDERLYING_LLM, variations=["gpt-4o-mini"])],
            "result": TestResult(
                evaluation_result=evaluation_result,
                conversation_history=conversation_history
            )
        }
    }
)