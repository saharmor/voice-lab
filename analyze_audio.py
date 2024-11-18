from pydub import AudioSegment, silence
from pyannote.audio import Pipeline
import torch

# Function to transcribe audio and get word timestamps
def transcribe_audio(audio_path):
    # Load pre-trained ASR model from SpeechBrain
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-en", savedir="tmpdir")
    
    # Transcribe the audio
    transcription = asr_model.transcribe_file(audio_path)
    return transcription

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
                                        use_auth_token="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")


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

# Example usage
audio_file_path = "test.wav"
speaker_segments = get_speaker_segments(audio_file_path)
pauses = detect_pauses(audio_file_path)

print("Detected Pauses with Speakers:")
for start, duration in pauses:
    if duration > 2:
        last_speaker = find_last_speaker(start, speaker_segments)
        print(f"Start: {start}s, Duration: {duration}s, Last Speaker: {last_speaker}")
