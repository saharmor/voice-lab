from dotenv import load_dotenv
from run_tests import generate_test_results_report, run_tests

if __name__ == "__main__":
    load_dotenv()
    test_result = run_tests(tests_to_run=["jailbreaking_airline_agent"])
    generate_test_results_report(test_result)