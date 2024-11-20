from typing import List
from llm_testing.core.data_types import EntitySpeaking
from ..data_types import CallSegment, PauseData

MIN_PAUSE_DURATION = 2

def detect_pauses(call_segments: List[CallSegment]) -> List[PauseData]:
    pauses = []
    prev_segment = call_segments[0]
    last_callee_segment = call_segments[0] if call_segments[0].speaker == EntitySpeaking.CALLEE else None
    # Check for pauses longer than min_pause_duration seconds between segments
    for i in range(1, len(call_segments)):
        current_segment = call_segments[i]
        
        if current_segment.speaker == EntitySpeaking.CALLEE and prev_segment.speaker == EntitySpeaking.VOICE_AGENT:
            last_callee_segment = current_segment

        # Only check pauses after callee segments
        if prev_segment.speaker == EntitySpeaking.CALLEE and current_segment.speaker == EntitySpeaking.VOICE_AGENT:
            if last_callee_segment:
                pause_duration = current_segment.start_time - last_callee_segment.end_time
                if pause_duration > MIN_PAUSE_DURATION:
                    pauses.append(PauseData(
                        duration=pause_duration,
                        start_time=prev_segment.end_time,
                        text_before_pause=prev_segment.text,
                        text_after_pause=current_segment.text,
                    ))
                    
                last_callee_segment = None

        prev_segment = current_segment

    return pauses
    