from torch.utils.data import Dataset
import io
from typing import Union, List

class AudioDataset(Dataset):
    def __init__(self, manifest: Union[str, List[str]]):
        super().__init__()