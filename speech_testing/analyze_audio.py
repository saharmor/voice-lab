import tempfile
import time
from pydub import AudioSegment, silence
import os
import stable_whisper
from dotenv import load_dotenv


load_dotenv()

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Function to detect silence and measure pauses
def detect_pauses(audio_path, silence_thresh=-50, min_silence_len=500):
    audio = AudioSegment.from_file(audio_path)
    
    # Detect silent chunks
    silent_chunks = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    # Convert silence start and end points to seconds
    pauses = [(start / 1000, (end - start) / 1000) for start, end in silent_chunks]
    return pauses

def detect_interuptions(results):
    interruption_segments = []
    prev_speaker = None
    prev_end_time = 0
    # Analyze segments to detect interpreter segments
    for res in results:
        current_speaker = res['speaker']

        if prev_speaker is not None:
            # Check if speaker has changed and if the start time is smaller or equal than last end time
            if current_speaker != prev_speaker and res['start_time'] <= prev_end_time:
                interruption_segments.append(res)

        # Update previous speaker and end time
        prev_speaker = current_speaker
        prev_end_time = res['end_time']

    print(f"\n\n***** Interruption segments: {len(interruption_segments)}")
    for res in interruption_segments:
        duration = res['end_time'] - res['start_time']
        # Find who was interrupted by looking at previous speaker
        interrupted_speaker = "Callee" if res['speaker'] == "Agent" else "Agent"
        if interrupted_speaker is not None:
            # print who interuppted whom and at what time
            print(f"Interruption at {res['start_time']:.2f}s (duration: {duration:.2f}s) - {interrupted_speaker} interrupted {res['speaker']}")
        print(f"Transcription: {res['text']}\n")

def transcribe_simple(model, audio_file_path: str):
    return model.transcribe(audio_file_path).to_dict()

def analyze_audio(audio_file_path: str, is_first_speaker_agent: bool = False):
    from pyannote.audio import Pipeline
    import torchaudio

    # Initialize the speaker diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="YOUR_HUGGINGFACE_TOKEN")

    # Perform speaker diarization
    print("Processing speaker diarization...")
    start_time = time.time()
    waveform, sample_rate = torchaudio.load(audio_file_path)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
    # diarization = pipeline(audio_file_path, num_speakers=2)
    #calcualte how much time it took
    end_time = time.time()
    print(f"Speaker diarization completed in {end_time - start_time:.2f} seconds")

    # List to store diarization results
    results = []

    model = stable_whisper.load_model('medium.en')
    speakers_mapping = {
        "SPEAKER_00": "Agent" if is_first_speaker_agent else "Callee",
        "SPEAKER_01": "Callee" if is_first_speaker_agent else "Agent"
    }
    # Iterate over each diarized segment
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
            # Perform speech recognition
            transcription = transcribe_simple(model, temp_file_path)

        results.append({
            'start_time': start_time,
            'end_time': end_time,
            'speaker': speakers_mapping[speaker],
            'text': transcription['text'].strip()
        })

    detect_interuptions(results)
    



audio_file_path = "interuption_tes.wav"
analyze_audio(audio_file_path)
# detect_interuptions(result)


# transcriber = WhisperTranscriber(model_size="large-v3")
# transcription = transcriber.transcribe(audio_file_path)


# speaker_segments = get_speaker_segments(audio_file_path)
# pauses = detect_pauses(audio_file_path)

# print("Detected Pauses with Speakers:")
# for start, duration in pauses:
#     if duration > 2:
#         last_speaker = find_last_speaker(start, speaker_segments)
#         print(f"Start: {start}s, Duration: {duration}s, Last Speaker: {last_speaker}")
