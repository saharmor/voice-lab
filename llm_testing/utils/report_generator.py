from typing import List, Dict, Any
from llm_testing.core.data_types import TestResult

def generate_report(results: List[TestResult]) -> Dict[str, Any]:
    """Generates a summary report of test results"""
    total_tests = len(results)
    passed_tests = sum(1 for result in results if result.passed)
    
    return {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
        "total_execution_time": sum(result.execution_time for result in results),
        "results": [
            {
                "test_name": result.test_case.name,
                "passed": result.passed,
                "execution_time": result.execution_time,
                "errors": result.errors
            }
            for result in results
        ]
    }
