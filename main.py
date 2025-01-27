from core.data_types import EvaluationResponse, MetricResult, TestResult
from core.utils.generate_report import generate_test_results_report
from speech_testing.run_tests import run_tests as run_speech_tests

from dotenv import load_dotenv

load_dotenv()

def suppress_output(all_output=False):
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

    if all_output:
        # Redirect stdout to suppress tqdm output
        import sys
        class DummyFile(object):
            def write(self, x): pass
            def flush(self): pass

        sys.stdout = DummyFile()  # Suppress tqdm
        tqdm.monitor_interval = 0  # Disable tqdm warning


suppress_output(all_output=False)

# Run text-based tests
# test_result = run_llm_tests()
# generate_test_results_report(test_result)

# Run speech-based tests
tests_result = run_speech_tests("speech_testing/audio_files",
                               "Qualify leads for a new voice agent called Jordan")
# tests_result = generate_mock_test_result()
# temp = determine_speakers(tests_result[0].call_segments, "Book a seat on a flight")


completed_tests = {}
for audio_file, test_result in tests_result.items():
    conversation_history = []
    for call_segment in test_result.call_segments:
        conversation_history.append({
            "speaker": call_segment.speaker.value,
            "text": call_segment.text,
            "start_timestamp": call_segment.start_time,
            "end_timestamp": call_segment.end_time
        })
    
    evaluation_result = EvaluationResponse(summary="mock summary", evaluation_results=[])
    evaluation_result.evaluation_results.append(
        MetricResult(name="interruptions", eval_output_type="success_flag",
                  eval_output="true" if len(test_result.interruptions) == 0 else "false",
                  eval_output_success_threshold=1,
                  reasoning=f"Had {len(test_result.interruptions)} interruptions.\n" + ("\n".join([f"\nInterruption at {i.interrupted_at:.2f}s:\nText that interrupted: {i.interruption_text}\n" for i in test_result.interruptions]) if test_result.interruptions else ""),
                  evidence="")
    )
    
    evaluation_result.evaluation_results.append(
        MetricResult(name="pauses", eval_output_type="success_flag",
                      eval_output="true" if len(test_result.pauses) == 0 else "false",
                      eval_output_success_threshold=1,
                      reasoning=f"Had {len(test_result.pauses)} pauses.\n" + ("\n".join([f"Pause at {p.start_time:.2f}s (duration: {p.duration:.2f}s). Text before pause: {p.text_before_pause}" for p in test_result.pauses]) if test_result.pauses else ""),
                      evidence="")
    )

    completed_tests[audio_file] = {
        "tested_component": [],
        "result": TestResult(
            evaluation_result=evaluation_result,
            conversation_history=conversation_history
        )
    }

generate_test_results_report(completed_tests)