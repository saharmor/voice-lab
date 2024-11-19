import os
from typing import List

from .data_types import CallSegment, Speaker, SpeechTestResult
from .data_types import CallSegment, Speaker, SpeechTestResult
from .metrics.interruptions import detect_interuptions
from .metrics.pauses import MIN_PAUSE_DURATION, detect_pauses
    
import tempfile
import time
from typing import List
from pyannote.audio import Pipeline
import torchaudio
import stable_whisper


def transcribe_simple(model, audio_file_path: str):
    return model.transcribe(audio_file_path).to_dict()

def transcribe_audio(audio_file_path: str, is_first_speaker_agent: bool) -> List[CallSegment]:
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    if not HUGGING_FACE_TOKEN:
        raise ValueError("Please set HUGGING_FACE_TOKEN environment variable")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGING_FACE_TOKEN)

    print("Performing speaker diarization...")
    start_time = time.time()
    waveform, sample_rate = torchaudio.load(audio_file_path)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
    end_time = time.time()
    print(f"--> ✨ Speaker diarization completed in {end_time - start_time:.2f} seconds")

    # List to store diarization results
    call_segments = []
    speakers_mapping = {
        "SPEAKER_00": Speaker.AGENT if is_first_speaker_agent else Speaker.CALLEE,
        "SPEAKER_01": Speaker.CALLEE if is_first_speaker_agent else Speaker.AGENT
    }
    model = stable_whisper.load_model('medium.en')
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_time = segment.start
        end_time = segment.end

        # Convert times to sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Extract the audio segment
        segment_waveform = waveform[:, start_sample:end_sample]
        
        # use temporary file to store segment
        transcription = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file: 
            temp_file_path = temp_file.name
            torchaudio.save(temp_file_path, segment_waveform, sample_rate)
            transcription = transcribe_simple(model, temp_file_path)

        call_segments.append(CallSegment(
            start_time=start_time,
            end_time=end_time,
            speaker=speakers_mapping[speaker],
            text=transcription['text'].strip()
        ))

def analyze_audio(audio_file_path: str, is_first_speaker_agent: bool = False, print_verbose: bool = False) -> SpeechTestResult:
    call_segments = transcribe_audio(audio_file_path, is_first_speaker_agent)
    interuptions = detect_interuptions(call_segments)
    pauses = detect_pauses(call_segments)

    if print_verbose:
        print(f"\n\n***** Detected {len(pauses)} long pauses (>{MIN_PAUSE_DURATION}s) after callee responses")
        for pause in pauses:
            print(f"Pause at {pause.start_time:.2f}s (duration: {pause.duration:.2f}s). Text before pause: {pause.text_before_pause}.")
    
        print(f"\n\n***** Interruption segments: {len(interuptions)}")
        for interruption in interuptions:
            print(f"Interruption at {interruption.interrupted_at:.2f}s (duration: {interruption.interruption_duration:.2f}s) - {interruption.interrupted_speaker.value} interrupted {interruption.interrupted_speaker.value}")
            print(f"Transcription: {interruption.interruption_text}\n")
            

    # TODO SpeechTestResult abstract - should be a list of test_type and test_result, later interpreted by whoever consumes it
    return SpeechTestResult(
        call_segments=call_segments,
        interruptions=interuptions,
        pauses=pauses,
    )



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
