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

from model_ds_v1_6 import TFiLMSuperResolution, create_tfilm_super_resolution

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
UPSCALE_FACTOR = 4
NUM_EPOCHS = 150
QUALITY_MODE = True # True if focusing on improving signal quality without changing length

MODEL_SAVE_NAME = "best_model_V1_6.pth"

# Initialize model in quality improvement mode since input/target have same length
model = create_tfilm_super_resolution(
    upscale_factor=UPSCALE_FACTOR,
    quality_mode=QUALITY_MODE,  
    base_channels=64,
    tfilm_hidden_size=128,
    block_size=256
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()

def prep_dataloaders(train_file_path, test_file_path):
    # Load data and print shapes for debugging
    X_train, y_train = load_h5(train_file_path)
    X_test, y_test = load_h5(test_file_path)

    # In VCTK dataset:
    # - 'data' contains low-resolution input (should be upsampled by model)
    # - 'label' contains high-resolution target (ground truth)
    print("\nLoaded data shapes:")
    print(f"Training - X (input_lr): {X_train.shape}, y (target_hr): {y_train.shape}")
    print(f"Testing  - X (input_lr): {X_test.shape}, y (target_hr): {y_test.shape}")

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

def train_model(model, num_epochs, optimizer, criterion, device, 
                train_loader, test_loader, best_test_loss, patience,
                best_model_path, epochs_no_improve):
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Run one training epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Run one testing (validation) epoch
        test_loss = test_epoch(model, test_loader, criterion, device)
        
        end_time = time.time()
        
        print(f"Epoch {epoch+1:02d}/{num_epochs:03d} | "
            f"Train Loss: {train_loss} | "
            f"Test Loss: {test_loss} | "
            f"Time: {end_time - start_time:.2f}s")

        # Check for improvement
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0

            # Save the best model checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fn': criterion.__class__.__name__,
                'best_test_loss': best_test_loss
            }
            torch.save(checkpoint, best_model_path)
            print(f"✅ New best model saved at epoch {epoch+1} with test loss {test_loss:.6f}")
            
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"⏹ Early stopping triggered after {epoch+1} epochs.")
            print(f"Best model saved as '{best_model_path}' with loss = {best_test_loss}")
            break
    
    return best_test_loss

print("Starting training...")
best_test_loss = float("inf")               # initialize to large value
patience = 20                               # stop if no improvement after N epochs
epochs_no_improve = 0                       # counter for early stopping
best_model_path = MODEL_SAVE_NAME

root_dir = Path(__file__).parent.parent  # Get project root directory

train_file_path_dir = root_dir / 'data' / 'vctk' / 'datasets'
# List of training file names
train_dataset_list = list(train_file_path_dir.glob('*vctk-speaker1-train.4.16000.8192.4096.h5'))

test_file_path_dir = root_dir / 'data' / 'vctk' / 'datasets' 
# List of test file names
val_dataset_list = list(test_file_path_dir.glob('*vctk-speaker1-val.4.16000.8192.4096.h5.tmp'))

for i, (train, val) in enumerate(zip(train_dataset_list, val_dataset_list)):
    print(f"Dataset {i+1}/{len(train_dataset_list)}")
    print(f"Training on: {train}")
    print(f"Validating on: {val}")
    
    t_loader, v_loader = prep_dataloaders(train, val)
    
    # Load previous best model if continuing training
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded previous best model checkpoint")

    best_test_loss = train_model(model, NUM_EPOCHS,
                                 optimizer, criterion, DEVICE,
                                 t_loader, v_loader,
                                 best_test_loss, patience,
                                 best_model_path, epochs_no_improve)

print(f"Best model saved as '{best_model_path}' with loss = {best_test_loss}")
summary(model)
print("Training finished.")
