import json
import os
from typing import Dict, List

from llm_testing.core.data_types import EntitySpeaking

from llm_testing.providers.openai import OpenAIProvider

from speech_testing.data_types import CallSegment, SpeechTestResult
from speech_testing.metrics.interruptions import detect_interuptions
from speech_testing.metrics.pauses import MIN_PAUSE_DURATION, detect_pauses
    
import time
from typing import List
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

import torchaudio
import stable_whisper


def transcribe_prescise_timestamps(model, audio_file_path: str):
    return model.refine(audio_file_path, model.transcribe(audio_file_path, suppress_silence=False)).to_dict()

def merge_diarization_and_transcription(diarization, transcription) -> List[CallSegment]:
    from pyannote.core import Segment
    import pandas as pd

    # Convert diarization result to a DataFrame for easier handling
    diarization_df = pd.DataFrame(
        [
            {
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker_label
            }
            for segment, _, speaker_label in diarization.itertracks(yield_label=True)
        ]
    )
    
    # Prepare a list to hold the final results
    final_transcriptions = []

    # Iterate over the words from the ASR result
    for segment in transcription["segments"]:
        for word_info in segment["words"]:
            word_start = word_info["start"]
            word_end = word_info["end"]
            
            # Find the speaker for this word
            # Check which diarization segment the word falls into
            speaker = None
            for idx, row in diarization_df.iterrows():
                if word_end > row["start"] and word_start < row["end"]:
                    speaker = row["speaker"]
                    break
        
        # If no speaker is found, label as 'Unknown' or skip
        if not speaker:
            speaker = "Unknown"

        segment_start = segment["start"]
        segment_end = segment["end"]
        segment_text = segment["text"]

        
        # Append to final transcriptions
        final_transcriptions.append(
            CallSegment(
                start_time=segment_start,
                end_time=segment_end,
                speaker=speaker,
                text=segment_text
            )
        )

    return final_transcriptions


def diarize_audio(audio_file_path: str) -> List[CallSegment]:
    api_key = os.getenv("HUGGING_FACE_TOKEN")
    if not api_key:
        raise ValueError("Please set HUGGING_FACE_TOKEN environment variable")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=api_key)
    # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=api_key)

    print("Performing speaker diarization...")
    start_time = time.time()
    waveform, sample_rate = torchaudio.load(audio_file_path)
    with ProgressHook() as hook:
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2, hook=hook)
    end_time = time.time()
    print(f"--> ✨ Speaker diarization completed in {end_time - start_time:.2f} seconds")
    
    return diarization


def determine_speakers(transcription: List[CallSegment], agent_task: str) -> Dict[str, EntitySpeaking]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    provider = OpenAIProvider(api_key, "gpt-4o")
    system_prompt = f'''I'm building a voice agent that calls people and businesses on my behalf. Here's a call transcript. Your role is to determine who is SPEAKER_00 and who is SPEAKER_01 by looking at the task I gave my voice agent and the transcript.
Return a json with the following format: {{"speaker_00": "callee" | "voice_agent", "speaker_01": "callee" | "voice_agent"}}
Return None if you cannot determine who is speaking or if there are more than 2 speakers.
DON'T RETURN ANYTHING ELSE BUT THE JSON. Format it correctly as I load it into a dictionary in python.
    
    # Task
    {agent_task}'''
    conversation_history = "\n".join([f"{segment.speaker}: {segment.text}" for segment in transcription])
    messages = [{"role": "user", "content": conversation_history}]
    response = provider.plain_call(system_prompt, messages)
    try:
        return json.loads(response.response_content)
    except json.JSONDecodeError:
        raise ValueError("Could not determine speakers - invalid JSON response")

def transcribe_audio(audio_file_path: str, agent_task: str) -> List[CallSegment]:#
    diarization = diarize_audio(audio_file_path)
    if not diarization:
        raise ValueError("No diarization results found")
    model = stable_whisper.load_model('large-v3-turbo')
    transcription = transcribe_prescise_timestamps(model, audio_file_path)
    diarizated_call_segments = merge_diarization_and_transcription(diarization, transcription)
    
    speakers_mapping = determine_speakers(diarizated_call_segments, agent_task)

    # fix speaker names
    for call_segment in diarizated_call_segments:
        call_segment.speaker = EntitySpeaking(speakers_mapping.get(call_segment.speaker.lower(), "unknown"))

    return diarizated_call_segments


def transcribe_using_assemblyai(audio_file_path: str, agent_task: str) -> List[CallSegment]:
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise ValueError("Please set ASSEMBLYAI_API_KEY environment variable")
    import assemblyai as aai

    aai.settings.api_key = api_key
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file_path, 
                                        config=aai.TranscriptionConfig(speaker_labels=True,
                                                                       speakers_expected=2,
                                                                       speech_model=aai.SpeechModel.nano,
                                                                       disfluencies=True))

    call_segments = []
    for utt in transcript.utterances:
        call_segments.append(CallSegment(
            start_time=utt.start / 1000,  # Convert from ms to seconds
            end_time=utt.end / 1000,
            speaker=utt.speaker,
            text=utt.text
        ))
    speakers_mapping = determine_speakers(call_segments, agent_task)

    # fix speaker names
    for call_segment in call_segments:
        call_segment.speaker = EntitySpeaking(speakers_mapping[call_segment.speaker])

    return call_segments

def analyze_audio(audio_file_path: str, agent_task: str, print_verbose: bool = False) -> SpeechTestResult:
    call_segments = transcribe_audio(audio_file_path, agent_task)
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



def run_tests(audio_files_dir: str, agent_task: str) -> Dict[str, SpeechTestResult]:
    # TODO Metrics should be intuerrptions, pauses, etc. Refactor accordingly
    test_number = 1
    tests_results = {}
    for audio_file in os.listdir(audio_files_dir):
        if not audio_file.startswith("11x_role"):
            continue

        print(f"\n\n=== Running speech test {test_number} of {len(os.listdir(audio_files_dir))} with [{audio_file}] ===")
        test_result = analyze_audio(os.path.join(audio_files_dir, audio_file), agent_task)   
        tests_results[audio_file] = test_result
        test_number += 1

        # TODO remove, just for faster debugging
        break

    print(f"\n\n=== All speech tests completed: {test_number - 1} ===")

    return tests_results
