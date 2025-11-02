import os
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import load_h5

class VctkWavDataset(Dataset):
    """Custom Dataset for lazy loading H5 files"""
    
    def __init__(self, list_files):
        self.list_files = sorted(list_files)
        # Preload all data for efficiency
        self._preload_data()  
    
    def _preload_data(self):
        """Preload all data from H5 files"""
        self.all_data = []
        self.all_labels = []
        
        for file_path in self.list_files:
            data, labels = load_h5(file_path)
            self.all_data.append(data)
            self.all_labels.append(labels)
        
        # Combine all data from different files
        self.all_data = np.concatenate(self.all_data, axis=0)
        self.all_labels = np.concatenate(self.all_labels, axis=0)
        
        print(f"Total dataset size: {self.all_data.shape}")
    
    def __len__(self):
        return len(self.all_data)  # Return total number of samples
    
    def __getitem__(self, idx):
        # Return single sample at index
        data_sample = self.all_data[idx]
        label_sample = self.all_labels[idx]
        return torch.from_numpy(data_sample).float(), torch.from_numpy(label_sample).float()
