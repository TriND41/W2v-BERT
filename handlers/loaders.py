from torch.utils.data import Dataset
import numpy as np
from resolvers.processor import AudioProcessor
import io
from typing import Union, List, Optional, Tuple

class AudioDataset(Dataset):
    def __init__(self, manifest: Union[str, List[str]], processor: AudioProcessor, num_examples: Optional[int] = None) -> None:
        super().__init__()
        if isinstance(manifest, str):
            self.paths = io.open(manifest).read().strip().split('\n')
        else:
            self.paths = manifest

        if num_examples is not None:
            self.paths = self.paths[:num_examples]

        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index: int) -> np.ndarray:
        audio_path = self.paths[index]
        audio = self.processor.load_audio(audio_path)
        return audio
    
    def collate(self, audios: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        waveforms, lengths = self.processor(audios)
        return waveforms, lengths