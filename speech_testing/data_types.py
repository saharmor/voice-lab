from enum import Enum

REQUIRED_AUDIO_TYPE = "float32"

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