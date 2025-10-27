import os
import numpy as np
from pathlib import Path
from utils import load_h5, upsample_wav, load_full_files
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from traintest import train_epoch, test_epoch
import time

from model_ds_v1_5 import TFiLMSuperResolution, create_tfilm_super_resolution
# from model_ds_v2 import TFiLMSuperResolution, create_tfilm_super_resolution

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
UPSCALE_FACTOR = 4
NUM_EPOCHS = 50
QUALITY_MODE = False # True if focusing on improving signal quality without changing length


root_dir = Path(__file__).parent.parent  # Get project root directory
train_file_path = root_dir / 'data' / 'vctk' / 'speaker1' / 'v_1_5_vctk-speaker1-train.4.16000.8192.4096.h5'      # Path to training data
test_file_path = root_dir / 'data' / 'vctk' / 'speaker1' / 'v_1_5_vctk-speaker1-val.4.16000.8192.4096.h5.tmp'     # Path to test data

# Custom loss that combines MSE with frequency domain loss
class SpectralLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.n_fft = 512
        self.hop_length = 256
        self.register_buffer('window', torch.hann_window(self.n_fft))
    
    def forward(self, y_pred, y_true):
        # Handle input shape (batch, seq_len, channels) -> (batch, channels, seq_len)
        if y_pred.shape[-1] == 1:
            y_pred = y_pred.transpose(1, 2)
            y_true = y_true.transpose(1, 2)
            
        # Time domain loss (MSE)
        mse_loss = F.mse_loss(y_pred, y_true)
        
        # Reshape for STFT: (batch, channels, seq_len) -> (batch * channels, seq_len)
        batch_size, channels, seq_len = y_pred.shape
        y_pred_flat = y_pred.reshape(-1, seq_len)
        y_true_flat = y_true.reshape(-1, seq_len)
        
        # Frequency domain loss with Hann window
        y_pred_fft = torch.stft(y_pred_flat, n_fft=self.n_fft, 
                               hop_length=self.hop_length,
                               win_length=self.n_fft,
                               window=self.window,
                               return_complex=True)
        y_true_fft = torch.stft(y_true_flat, n_fft=self.n_fft,
                               hop_length=self.hop_length,
                               win_length=self.n_fft,
                               window=self.window,
                               return_complex=True)
        
        # Magnitude loss in frequency domain
        spec_loss = F.mse_loss(y_pred_fft.abs(), y_true_fft.abs())
        
        # Combined loss with logging
        loss = (1 - self.alpha) * mse_loss + self.alpha * spec_loss
        if torch.isnan(loss).any():
            print(f"WARNING: NaN in loss. MSE: {mse_loss:.3e}, Spec: {spec_loss:.3e}")
            return mse_loss  # Fallback to just MSE if spectral loss fails
            
        return loss

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
        print(f"Input {in_len} and target {tgt_len} same length. Model will learn to improve quality while preserving length.")
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
    quality_mode=QUALITY_MODE,  
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
print(f"Training - X (input_lr): {X_train.shape}, y (target_hr): {y_train.shape}")
print(f"Testing  - X (input_lr): {X_test.shape}, y (target_hr): {y_test.shape}")

#X_train, y_train = _ensure_input_target(X_train, y_train, UPSCALE_FACTOR)
#X_test, y_test = _ensure_input_target(X_test, y_test, UPSCALE_FACTOR)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) #, weight_decay=1e-5)
criterion = nn.MSELoss()
#criterion = SpectralLoss(alpha=0.8)  # Increase spectral loss weight

num_epochs = NUM_EPOCHS

print("Starting training...")

for epoch in range(num_epochs):
    start_time = time.time()
    
    # Run one training epoch
    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    
    # Run one testing (validation) epoch
    test_loss = test_epoch(model, test_loader, criterion, DEVICE)
    
    end_time = time.time()
    
    print(f"Epoch {epoch+1:02d}/{num_epochs:03d} | "
          f"Train Loss: {train_loss} | "
          f"Test Loss: {test_loss} | "
          f"Time: {end_time - start_time:.2f}s")
    
    if epoch >= NUM_EPOCHS - 1:
        print(f"Training finished. {epoch + 1} epochs completed")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion.state_dict(),
        }
        torch.save(checkpoint, f'checkpoint_V_1_5_{epoch+1}.pth')


summary(model)