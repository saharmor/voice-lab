import tempfile
import time
from pydub import AudioSegment, silence
from pyannote.audio import Pipeline
import os
import stable_whisper
from dotenv import load_dotenv

from transcribe import WhisperTranscriber

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

def get_speaker_segments(audio_path):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token=HUGGING_FACE_TOKEN)


    # apply the pipeline to an audio file
    diarization = pipeline(audio_path, num_speakers=2)

    # dump the diarization output to disk using RTTM format
    with open("audio.rttm", "w") as rttm:
        diarization.write_rttm(rttm)
    
    
    # Convert to list of (start, end, speaker) tuples
    segments = [(segment.start, segment.end, speaker) 
                for segment, _, speaker in diarization.itertracks()]
    return segments

def find_last_speaker(time_point, speaker_segments):
    last_speaker = None
    for start, end, speaker in speaker_segments:
        if start <= time_point <= end:
            return speaker
        if end < time_point:
            last_speaker = speaker
    return last_speaker

def load_audio_file(audio_file_path: str):
    # Initialize the speaker diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGING_FACE_TOKEN)
    # Perform speaker diarization
    diarization = pipeline(audio_file_path)

    # List to store diarization results
    results = []

    # Iterate over each diarized segment
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_time = segment.start
        end_time = segment.end
        
        # Store the result
        results.append({
            'content': segment.label,
            'start_time': start_time,
            'end_time': end_time,
            'speaker': speaker,
        })

    return results

def detect_interuptions(results: list[dict]):
    '''
        results = [{
            'start_time': start_time,
            'end_time': end_time,
            'speaker': speaker
    }]
    '''

    interpreter_segments = []
    prev_speaker = None

    # Analyze segments to detect interpreter segments
    for res in results:
        current_speaker = res['speaker']
        
        if prev_speaker is not None:
            # Check if speaker has changed
            if current_speaker != prev_speaker:
                interpreter_segments.append(res)
        
        # Update previous speaker and language
        prev_speaker = current_speaker

    # Output interpreter segments
    for res in interpreter_segments:
        print(f"Interpreter segment: {res['start_time']:.2f}s - {res['end_time']:.2f}s")
        print(f"Speaker: {res['speaker']}, Language: {res['language']}\n")


def test_full(audio_file_path: str, is_first_speaker_agent: bool = False):
    import logging

    # Set logging level to WARNING to suppress INFO messages
    logging.getLogger('speechbrain').setLevel(logging.ERROR)
    logging.getLogger('pyannote.audio').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('speechbrain.utils').setLevel(logging.ERROR)

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
            transcription = transcribe_simple(temp_file_path)

        # Store the result
        results.append({
            'start_time': start_time,
            'end_time': end_time,
            'speaker': "Agent" if is_first_speaker_agent and speaker == 'SPEAKER_00' else "Callee",
            'text': transcription['text'].strip()
        })

    # Initialize variables for analysis
    interpreter_segments = []
    prev_speaker = None
    prev_end_time = 0
    # Analyze segments to detect interpreter segments
    for res in results:
        current_speaker = res['speaker']

        if prev_speaker is not None:
            # Check if speaker has changed and if the start time is smaller or equal than last end time
            if current_speaker != prev_speaker and res['start_time'] <= prev_end_time:
                interpreter_segments.append(res)

        # Update previous speaker and end time
        prev_speaker = current_speaker
        prev_end_time = res['end_time']

    print(f"\n\n*****Interpreter segments: {len(interpreter_segments)}")
    for res in interpreter_segments:
        duration = res['end_time'] - res['start_time']
        # Find who was interrupted by looking at previous speaker
        interrupted_speaker = prev_speaker if prev_speaker != res['speaker'] else None
        if interrupted_speaker is not None:
            # print who interuppted whom and at what time
            print(f"Interruption at {res['start_time']:.2f}s (duration: {duration:.2f}s) - {interrupted_speaker} interrupted {res['speaker']}")
        print(f"Transcription: {res['text']}\n")

def transcribe_simple(audio_file_path: str):
    model = stable_whisper.load_model('medium.en')
    result = model.transcribe(audio_file_path)
    return result.to_dict()


audio_file_path = "interuption_tes.wav"
test_full(audio_file_path)
# detect_interuptions(result)


transcriber = WhisperTranscriber(model_size="large-v3")
transcription = transcriber.transcribe(audio_file_path)


speaker_segments = get_speaker_segments(audio_file_path)
pauses = detect_pauses(audio_file_path)

print("Detected Pauses with Speakers:")
for start, duration in pauses:
    if duration > 2:
        last_speaker = find_last_speaker(start, speaker_segments)
        print(f"Start: {start}s, Duration: {duration}s, Last Speaker: {last_speaker}")
