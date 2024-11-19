from dataclasses import dataclass
from enum import Enum
from typing import List

class Speaker(Enum):
    AGENT = "Agent"
    CALLEE = "Callee"

@dataclass
class CallSegment:
    start_time: float
    end_time: float
    speaker: str
    text: str


class WhisperModelSize(Enum):
    TINY = 'tiny'
    TINY_ENGLISH = 'tiny.en'
    BASE = 'base'
    BASE_ENGLISH = 'base.en'
    SMALL = 'small'
    SMALL_ENGLISH = 'small.en'
    MEDIUM = 'medium'
    MEDIUM_ENGLISH = 'medium.en'
    LARGE_V1 = 'large-v1'
    LARGE_V2 = 'large-v2'
    LARGE_V3 = 'large-v3'

SPEAKER_MAPPING = {
    -1: "unknown"
}

@dataclass
class InterruptionData:
    interrupted_speaker: Speaker
    interrupted_at: float
    interruption_duration: float
    interruption_text: str

@dataclass
class PauseData:
    duration: float
    start_time: float
    text_before_pause: str
    text_after_pause: str

@dataclass
class SpeechTestResult:
    interruptions: List[InterruptionData]
    pauses: List[PauseData]
