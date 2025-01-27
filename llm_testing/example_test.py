from dotenv import load_dotenv
from .run_tests import run_tests
from core.utils.generate_report import generate_test_results_report

if __name__ == "__main__":
    load_dotenv()
    # test_result = run_tests(tests_to_run=["jailbreaking_airline_agent"])
    test_result = run_tests()
    generate_test_results_report(test_result)