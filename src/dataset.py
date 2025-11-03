import os
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import load_h5

class VctkWavDataset(Dataset):
    """Custom Dataset for lazy loading H5 files"""
    
    def __init__(self, list_files):
        self.list_files = sorted(list_files)
        self.samples = []
        
        # Pre-load sample indices
        for file_path in self.list_files:
            data, labels = load_h5(file_path)
            for i in range(len(data)):
                self.samples.append((file_path, i))

    def __len__(self):
        return len(self.samples) # Return total number of recordings

    def __getitem__(self, idx):
        # Return single recording from list of files
        file_path, sample_idx = self.samples[idx]
        data, labels = load_h5(file_path)
        
        # Return single sample, not the entire dataset
        return torch.from_numpy(data[sample_idx]).float(), torch.from_numpy(labels[sample_idx]).float()
