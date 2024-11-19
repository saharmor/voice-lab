from dotenv import load_dotenv
from .run_tests import run_tests


if __name__ == "__main__":
    load_dotenv()
    test_result = run_tests("audio_files")