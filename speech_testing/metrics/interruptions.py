from typing import List
from speech_testing.data_types import CallSegment, InterruptionData, Speaker


def detect_interuptions(call_segments: List[CallSegment]) -> List[InterruptionData]:
    interruption_segments = []
    prev_speaker = None
    prev_end_time = 0
    # Analyze segments to detect interruption segments
    for res in call_segments:
        current_speaker = res.speaker

        if prev_speaker is not None:
            # Check if speaker has changed and if the start time is smaller or equal than last end time
            if current_speaker != prev_speaker and res.start_time <= prev_end_time:
                interruption_segments.append(res)

        # Update previous speaker and end time
        prev_speaker = current_speaker
        prev_end_time = res.end_time

    interruption_data = []
    for res in interruption_segments:
        duration = res.end_time - res.start_time
        # Find who was interrupted by looking at previous speaker
        interrupted_speaker = Speaker.AGENT if res.speaker == Speaker.CALLEE else Speaker.CALLEE    
        interruption_data.append(InterruptionData(  
            interrupted_speaker=interrupted_speaker,
            interrupted_at=res.start_time,
            interruption_duration=duration,
            interruption_text=res.text
        ))

    return interruption_data