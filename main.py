from llm_testing.run_tests import run_tests as run_llm_tests
from llm_testing.utils.generate_report import generate_test_results_report
from speech_testing.run_tests import run_tests as run_speech_tests

from dotenv import load_dotenv

load_dotenv()

# Run text-based tests
# test_result = run_llm_tests()
# generate_test_results_report(test_result)

# Run speech-based tests
test_result = run_speech_tests(["interruption_test.wav"])
generate_test_results_report(test_result)
