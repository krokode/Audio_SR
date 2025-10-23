import os
import numpy as np
from pathlib import Path
from utils import load_h5, upsample_wav, load_full_files
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from traintest import train_epoch, test_epoch
import time

# from model_ds_v1 import TFiLMSuperResolution, create_tfilm_super_resolution
from model_ds_v2 import TFiLMSuperResolution, create_tfilm_super_resolution

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
UPSCALE_FACTOR = 4

root_dir = Path(__file__).parent.parent  # Get project root directory
train_file_path = root_dir / 'data' / 'vctk' / 'speaker1' / 'vctk-speaker1-train.4.16000.8192.4096.h5'      # Path to training data
test_file_path = root_dir / 'data' / 'vctk' / 'speaker1' / 'vctk-speaker1-val.4.16000.8192.4096.h5.tmp'     # Path to test data

def _ensure_input_target(X, Y, upscale):
    """Validate and prepare input/target pairs for super-resolution training.
    X should be the low-res input, Y the high-res target."""
    in_len = X.shape[1]
    tgt_len = Y.shape[1]
    
    # Case 1: Input needs to be upsampled to match target (normal case)
    if in_len * upscale == tgt_len:
        print("Input will be upsampled to match target resolution")
        return X, Y
        
    # Case 2: Input and target are same length (target needs interpolation)
    elif in_len == tgt_len:
        print("Input and target same length. Model will learn to improve quality while preserving length.")
        return X, Y
        
    # Case 3: Check if data/label are swapped
    elif Y.shape[1] * upscale == X.shape[1]:
        print("Detected reversed input/target in HDF5. Swapping X and Y.")
        return Y, X
        
    # Error case: Incompatible shapes
    raise RuntimeError(
        f"Input/target lengths incompatible with upscale={upscale}:\n"
        f"input_len={in_len}, target_len={tgt_len}\n"
        f"Expected either:\n"
        f"1. target_len = input_len * {upscale} (for normal upsampling), or\n"
        f"2. target_len = input_len (for quality improvement)"
    )


# Create model in quality improvement mode since input/target have same length
model = create_tfilm_super_resolution(
    upscale_factor=UPSCALE_FACTOR,
    quality_mode=True,  # Focus on improving signal quality without changing length
    base_channels=64,
    tfilm_hidden_size=128,
    block_size=256
).to(DEVICE)

# summary(model)

# Load data and print shapes for debugging
X_train, y_train = load_h5(train_file_path)
X_test, y_test = load_h5(test_file_path)

# In VCTK dataset:
# - 'data' contains low-resolution input (should be upsampled by model)
# - 'label' contains high-resolution target (ground truth)
print("\nLoaded data shapes:")
print(f"Training - X (input): {X_train.shape}, y (target): {y_train.shape}")
print(f"Testing  - X (input): {X_test.shape}, y (target): {y_test.shape}")

X_train, y_train = _ensure_input_target(X_train, y_train, UPSCALE_FACTOR)
X_test, y_test = _ensure_input_target(X_test, y_test, UPSCALE_FACTOR)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()

num_epochs = 1

print("Starting training...")

for epoch in range(num_epochs):
    start_time = time.time()
    
    # Run one training epoch
    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    
    # Run one testing (validation) epoch
    test_loss = test_epoch(model, test_loader, criterion, DEVICE)
    
    end_time = time.time()
    
    print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Test Loss: {test_loss:.4f} | "
          f"Time: {end_time - start_time:.2f}s")

print("Training finished.")

# summary(model)

# (Optional: save the model)
torch.save(model.state_dict(), 'tfilm_superres_quality_model.pth')