import os
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import load_h5

class VctkWavDataset(Dataset):
    """Custom Dataset for lazy loading H5 files"""
    def __init__(self, list_files):
        self.list_files = sorted(list_files)

    def __len__(self):
        return len(self.list_files) # Return total number of recordings

    def __getitem__(self, idx):
        # Return single recording from list of files
        data, labels = load_h5(self.list_files[idx])
        # Ensure correct shape: (channels, length)
        data = data.T if data.shape[0] > data.shape[1] else data
        labels = labels.T if labels.shape[0] > labels.shape[1] else labels
        return torch.from_numpy(data).float(), torch.from_numpy(labels).float()