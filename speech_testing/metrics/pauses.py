from typing import List
from speech_testing.data_types import CallSegment, PauseData, Speaker

MIN_PAUSE_DURATION = 3

def detect_pauses(call_segments: List[CallSegment]) -> List[PauseData]:
    pauses = []
    prev_segment = call_segments[0]
    # Check for pauses longer than min_pause_duration seconds between segments
    for i in range(1, len(call_segments)):
        current_segment = call_segments[i]
        
        # Only check pauses after callee segments
        if prev_segment.speaker == Speaker.CALLEE and current_segment.speaker == Speaker.AGENT:
            pause_duration = current_segment.start_time - prev_segment.end_time
            if pause_duration > MIN_PAUSE_DURATION:
                pauses.append(PauseData(
                    duration=pause_duration,
                    start_time=prev_segment.end_time,
                    text_before_pause=prev_segment.text,
                    text_after_pause=current_segment.text,
                ))

        prev_segment = current_segment

    return pauses
    