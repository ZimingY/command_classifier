import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from params import *

MAP = {'zero':0,
        'one':1,
        'two':2,
        'three':3,
        'four':4,
        'five':5,
        'six':6,
        'seven':7,
        'eight':8,
        'nine':9}
LABELS = {v:k for k, v in MAP.items()}

class AudioDataset(Dataset):
    """speech dataste"""

    def __init__(self, data_path, split):
        self.audio = []
        self.labels = []
        for i in range(num_classes):
            file_name = os.path.join(data_path, split, LABELS[i], f'{split}.npy')
            data = np.load(file_name)
            self.audio.append(data)
            self.labels.extend([i]*data.shape[0])
        self.audio = torch.from_numpy(np.concatenate(self.audio, axis=0))
        self.labels = torch.tensor(self.labels)
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.audio[idx], self.labels[idx]

