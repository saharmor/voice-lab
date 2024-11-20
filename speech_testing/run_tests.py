import json
import os
from typing import Dict, List

from llm_testing.core.data_types import EntitySpeaking

from llm_testing.providers.openai import OpenAIProvider

from speech_testing.data_types import CallSegment, SpeechTestResult
from speech_testing.metrics.interruptions import detect_interuptions
from speech_testing.metrics.pauses import MIN_PAUSE_DURATION, detect_pauses
    
import tempfile
import time
from typing import List
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

import torchaudio
import stable_whisper


def transcribe_simple(model, audio_file_path: str):
    return model.transcribe(audio_file_path).to_dict()

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
    call_segments = []
    model = stable_whisper.load_model('large-v3-turbo')
    waveform, sample_rate = torchaudio.load(audio_file_path)
    diarization = diarize_audio(audio_file_path)
    if not diarization:
        raise ValueError("No diarization results found")
    
    print(f"About to process {len(diarization)} segments")
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

        if transcription:
            call_segments.append(CallSegment(
                start_time=start_time,
                end_time=end_time,
                speaker=speaker,
                text=transcription['text'].strip()
            ))
    
    speakers_mapping = determine_speakers(call_segments, agent_task)

    # fix speaker names
    for call_segment in call_segments:
        call_segment.speaker = speakers_mapping[call_segment.speaker]

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
        if not audio_file.startswith("11x"):
            continue

        print(f"\n\n=== Running speech test {test_number} of {len(os.listdir(audio_files_dir))} with [{audio_file}] ===")
        test_result = analyze_audio(os.path.join(audio_files_dir, audio_file), agent_task)   
        tests_results[audio_file] = test_result
        test_number += 1

        # TODO remove, just for faster debugging
        break

    print(f"\n\n=== All speech tests completed: {test_number - 1} ===")

    return tests_results
