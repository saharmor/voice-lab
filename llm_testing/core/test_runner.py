from typing import List, Dict, Any
from datetime import datetime
import time
from llm_testing.core.data_types import TestCase, TestResult
from llm_testing.core.interfaces import LLMInterface

class LLMTestRunner:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.results: List[TestResult] = []
    
    def run_test(self, test_case: TestCase) -> TestResult:
        start_time = time.time()
        
        try:
            response = self.llm.generate_response(
                test_case.context,
                test_case.user_input
            )
            
            errors = test_case.expectations.validate(response)
            
            result = TestResult(
                test_case=test_case,
                response=response,
                passed=len(errors) == 0,
                errors=errors,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            result = TestResult(
                test_case=test_case,
                response="",
                passed=False,
                errors=[f"Execution error: {str(e)}"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        return result

    def run_test_suite(self, test_cases: List[TestCase]) -> List[TestResult]:
        """Runs multiple test cases and returns all results"""
        return [self.run_test(test_case) for test_case in test_cases]

