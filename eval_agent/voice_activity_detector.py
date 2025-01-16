import numpy as np
import torch


class VoiceActivityDetector:
    def __init__(self):
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                     model='silero_vad',
                                     force_reload=False)
        self.model.eval()

    def is_speech(self, audio_data: bytes, threshold: float = 0.5) -> bool:
        # Convert raw bytes directly to numpy array of int16
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Convert to float32 and normalize to [-1, 1]
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_float)
        
        # Get speech probability
        speech_prob = self.model(audio_tensor, 16000).item()
        return speech_prob > threshold  # Adjust threshold as needed