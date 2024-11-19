import os
from typing import List

from speech_testing.data_types import CallSegment, InterruptionData, PauseData, Speaker, SpeechTestResult

from .analyze_audio import analyze_audio
    
def run_tests(audio_files: List[str]):
    api_key = os.getenv("HUGGING_FACE_TOKEN")
    if not api_key:
        raise ValueError("Please set HUGGING_FACE_TOKEN environment variable")

    # TODO tests should be intuerrptions, pauses, etc. Refactor accordingly
    test_number = 1
    tests_results = []
    for audio_file in audio_files:
        print(f"\n\n=== Running speech test {test_number} of {len(audio_files)}: {audio_file} ===")
        test_result = analyze_audio(audio_file)   
        tests_results.append(test_result)
        test_number += 1

    print(f"\n\n=== All speech tests completed: {test_number - 1} ===")

    return tests_results
