import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
import time

from model_ds_v1_6 import TFiLMSuperResolution, create_tfilm_super_resolution
from traintest import train_epoch, test_epoch
from utils import load_h5, upsample_wav, load_full_files
from dataset import VctkWavDataset


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128 # depending on available mamory choose what batch size to use 1, 2, 4, 8, 16, 32, 64, 128
    upscale_factor = 4
    num_epochs = 150
    quality_mode = True 
    # True if focusing on improving signal quality without changing length
    # To use quality_mode = False use upscaled targets 32768 instead of 8192
    # *vctk-speaker1-train.4.16000.32768.4096.h5' and *vctk-speaker1-val.4.16000.32768.4096.h5.tmp' h5 files
    
    MODEL_TMP = lambda: "model_tmp.pth"
    MODEL_BEST = lambda: "model_best.pth"
    MODEL_CHECKPOINT = lambda: "some_checkpoint_to_resume_from.pth"
    
    # Initialize model in quality improvement mode since input/target have same length
    model = create_tfilm_super_resolution(
        upscale_factor=upscale_factor,
        quality_mode=quality_mode,  
        base_channels=64,
        tfilm_hidden_size=128,
        block_size=256
    ).to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    print("Starting training...")
    epochs_no_improve = 0 # counter for early stopping
    best_train_loss = float("inf") # initialize to large value
    best_test_loss = float("inf") # initialize to large value
    patience = 20 # stop if no improvement after N epochs

    root_dir = Path(__file__).parent.parent  # Get project root directory
    train_file_path_dir = root_dir / 'data' / 'vctk' / 'datasets'
    train_dataset_list = list(train_file_path_dir.glob('*vctk-speaker1-train.4.16000.8192.4096.h5'))

    test_file_path_dir = root_dir / 'data' / 'vctk' / 'datasets' 
    test_dataset_list = list(test_file_path_dir.glob('*vctk-speaker1-val.4.16000.8192.4096.h5.tmp'))

    # NOTE: will start from the existing checkpoint if `MODEL_SAVE_NAME` exists
    # Load previous best model if continuing training from a saved interrupted checkpoint.
    if os.path.exists(MODEL_CHECKPOINT()):
        checkpoint = torch.load(MODEL_CHECKPOINT(), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_train_loss = checkpoint['train_loss']
        best_test_loss = checkpoint['test_loss']
        print("Loaded from checkpoint")

    train_dataset = VctkWavDataset(train_dataset_list)
    test_dataset = VctkWavDataset(test_dataset_list)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        # Run one training epoch and step the optimizer
        start_time_train = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device=device)
        end_time_train = time.time()

        # Run one testing (validation) epoch
        start_time_test = time.time()
        test_loss = test_epoch(model, test_loader, criterion, device=device)
        end_time_test = time.time()

        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss} | "
            f"Test Loss: {test_loss} | "
            f"Train time: {end_time_train - start_time_train:.2f}s"
            f"Test time: {end_time_test - start_time_test:.2f}s")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_fn': criterion.__class__.__name__,
            'train_loss': train_loss,
            'test_loss': test_loss
        }

        torch.save(checkpoint, MODEL_TMP())

        # Check for improvement
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0

            # Save the best model checkpoint
            torch.save(checkpoint, MODEL_BEST())
            print(f"✅ New best at epoch {epoch+1}\nTrain: {train_loss:.6f}\nTest: {test_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"⏹ Early stopping triggered after {epoch+1} epochs.")
            print(f"Best model saved as '{MODEL_BEST()}' with loss = {best_test_loss}")
            break

    print(f"Best model saved as '{MODEL_BEST()}'\nTrain loss: {best_train_loss}\nTest loss: {best_test_loss}")
    print("Experiment finished.")
