import numpy as np
import librosa
from typing import List, Tuple

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def load_audio(self, path: str) -> np.ndarray:
        signal, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        return signal
    
    def __call__(self, audios: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        lengths = []
        max_length = 0

        for audio in audios:
            length = len(audio)
            if max_length < length:
                lengths.append(length)

        padded_audios = []
        for index, audio in enumerate(audios):
            padded_audios.append(
                np.pad(
                    array=audio,
                    pad_width=[0, max_length - lengths[index]],
                    mode='constant',
                    constant_values=0.0
                )
            )

        return np.stack(padded_audios, axis=0).astype(np.float32), np.array(lengths, dtype=np.int32)