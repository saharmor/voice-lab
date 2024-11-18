from dataclasses import dataclass
from enum import Enum
import tempfile
import time
from typing import List
from pyannote.audio import Pipeline
import torchaudio
import stable_whisper
import os
from dotenv import load_dotenv

class Speaker(Enum):
    AGENT = "Agent"
    CALLEE = "Callee"

@dataclass
class CallSegment:
    start_time: float
    end_time: float
    speaker: str
    text: str


load_dotenv()

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


def detect_pauses(call_segments: List[CallSegment], min_pause_duration: float = 1.5):
    pauses = []
    prev_segment = call_segments[0]
    # Check for pauses longer than min_pause_duration seconds between segments
    for i in range(1, len(call_segments)):
        current_segment = call_segments[i]
        
        # Only check pauses after callee segments
        if prev_segment.speaker == Speaker.CALLEE and current_segment.speaker == Speaker.AGENT:
            pause_duration = current_segment.start_time - prev_segment.end_time
            if pause_duration > min_pause_duration:
                pauses.append({
                    'duration': pause_duration,
                    'start_time': prev_segment.end_time,
                    'text_before_pause': prev_segment.text,
                    'text_after_pause': current_segment.text,
                })

        prev_segment = current_segment

    # Print detected pauses
    print(f"\n\n***** Detected {len(pauses)} long pauses (>{min_pause_duration}s) after agent responses")
    for pause in pauses:
        print(f"Pause at {pause['start_time']:.2f}s (duration: {pause['duration']:.2f}s). Text before pause: {pause['text_before_pause']}.")
    
    return pauses
    
def detect_interuptions(results):
    interruption_segments = []
    prev_speaker = None
    prev_end_time = 0
    # Analyze segments to detect interruption segments
    for res in results:
        current_speaker = res.speaker

        if prev_speaker is not None:
            # Check if speaker has changed and if the start time is smaller or equal than last end time
            if current_speaker != prev_speaker and res.start_time <= prev_end_time:
                interruption_segments.append(res)

        # Update previous speaker and end time
        prev_speaker = current_speaker
        prev_end_time = res.end_time

    print(f"\n\n***** Interruption segments: {len(interruption_segments)}")
    for res in interruption_segments:
        duration = res.end_time - res.start_time
        # Find who was interrupted by looking at previous speaker
        interrupted_speaker = Speaker.AGENT if res.speaker == Speaker.CALLEE else Speaker.CALLEE    
        print(f"Interruption at {res.start_time:.2f}s (duration: {duration:.2f}s) - {res.speaker.value} interrupted {interrupted_speaker.value}")
        print(f"Transcription: {res.text}\n")

def transcribe_simple(model, audio_file_path: str):
    return model.transcribe(audio_file_path).to_dict()

def analyze_audio(audio_file_path: str, is_first_speaker_agent: bool = False):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="YOUR_HUGGINGFACE_TOKEN")

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

    interuptions = detect_interuptions(call_segments)
    pauses = detect_pauses(call_segments)


# audio_file_path = "pauses_test.wav"
audio_file_path = "interruption_test.wav"
analyze_audio(audio_file_path)
# pauses = detect_pauses(audio_file_path)