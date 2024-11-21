from typing import List
from llm_testing.core.data_types import EntitySpeaking
from ..data_types import CallSegment, InterruptionData

# approach one - split into two channels, then run analysis (pauses + interruptions) on each channel
# approach two - speaker diarization on combined audio, then run analysis (pauses + intterruptions)

def detect_interuptions(call_segments: List[CallSegment]) -> List[InterruptionData]:
    interruption_segments = []
    prev_speaker = None
    prev_end_time = 0
    # Analyze segments to detect interruption segments
    for res in call_segments:
        current_speaker = res.speaker

        if prev_speaker is not None:
            # Check if speaker has changed and if the initial speaker's speech start time is smaller or equal than last end time
            if current_speaker != prev_speaker:
                # Only update prev_end_time when speaker changes
                if res.start_time <= prev_end_time:
                    interruption_segments.append(res)
                prev_end_time = res.end_time

        # Only update previous speaker
        prev_speaker = current_speaker

    interruption_data = []
    for res in interruption_segments:
        duration = res.end_time - res.start_time
        # Find who was interrupted by looking at previous speaker
        interrupted_speaker = EntitySpeaking.VOICE_AGENT if res.speaker == EntitySpeaking.CALLEE else EntitySpeaking.CALLEE    
        interruption_data.append(InterruptionData(  
            interrupted_speaker=interrupted_speaker,
            interrupted_at=res.start_time,
            interruption_duration=duration,
            interruption_text=res.text
        ))

    return interruption_data