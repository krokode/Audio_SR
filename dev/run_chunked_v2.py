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
CHUNK_SIZE = 8192  # Base chunk size for input sequences
NUM_EPOCHS = 50

print(f"Using device: {DEVICE}")

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
        print("Detected reversed input/target in data. Swapping X and Y.")
        return Y, X
    
    # Error case: Incompatible shapes
    raise RuntimeError(
        f"Input/target lengths incompatible with upscale={upscale}:\n"
        f"input_len={in_len}, target_len={tgt_len}\n"
        f"Expected either:\n"
        f"1. target_len = input_len * {upscale} (for normal upsampling), or\n"
        f"2. target_len = input_len (for quality improvement)"
    )

def normalize_audio(data):
    """Normalize audio data to [-1, 1] range"""
    if isinstance(data, list):
        # Handle list of sequences
        normalized = []
        for seq in data:
            seq = np.array(seq)
            max_val = np.abs(seq).max()
            if max_val > 0:
                normalized.append(seq / max_val)
            else:
                normalized.append(seq)
        return normalized
    else:
        # Handle numpy array
        max_val = np.abs(data).max()
        if max_val > 0:
            return data / max_val
        return data

# Create model in super-resolution mode for upscaling
model = create_tfilm_super_resolution(
    upscale_factor=UPSCALE_FACTOR,
    quality_mode=False,  # Perform audio super-resolution with upscaling
    base_channels=64,
    tfilm_hidden_size=128,
    block_size=256
).to(DEVICE)

# summary(model)

# Load and normalize full data files
X_train, y_train, X_test, y_test = load_full_files()
X_train = normalize_audio(X_train)
y_train = normalize_audio(y_train)
X_test = normalize_audio(X_test)
y_test = normalize_audio(y_test)

# Print raw data info
print("\nRaw data info:")
print(f"X_train type: {type(X_train)}, length: {len(X_train)}")
print(f"Sample X_train shape: {np.array(X_train[0]).shape}")
print(f"y_train type: {type(y_train)}, length: {len(y_train)}")
print(f"Sample y_train shape: {np.array(y_train[0]).shape}")

# Add data augmentation
def augment_chunk(chunk):
    # Add small random noise
    noise = np.random.normal(0, 0.001, chunk.shape)
    return chunk + noise

def process_sequences_to_chunks(sequences, chunk_size=CHUNK_SIZE):
    """Process sequences into fixed-size chunks.
    Creates input (downsampled) and target (original) chunks with proper upscaling relationship."""
    input_chunks = []
    target_chunks = []
    skipped_sequences = 0
    total_sequences = len(sequences)
    
    print(f"\nAnalyzing sequences:")
    print(f"Total sequences: {total_sequences}")
    print(f"Input chunk size: {chunk_size}")
    print(f"Target chunk size: {chunk_size * UPSCALE_FACTOR}")
    
    # Print some sequence statistics
    sequence_lengths = [len(seq) for seq in sequences]
    print(f"\nSequence length stats:")
    print(f"Min: {min(sequence_lengths)}, Max: {max(sequence_lengths)}")
    print(f"Mean: {np.mean(sequence_lengths):.1f}")
    
    for idx, seq in enumerate(tqdm(sequences, desc="Processing sequences")):
        # Convert to float32 and reshape if needed
        seq = np.array(seq, dtype=np.float32)
        if seq.ndim == 2:  # If shape is (length, 1)
            seq = seq.reshape(-1)
        
        # Skip if sequence is too short
        if len(seq) < chunk_size * UPSCALE_FACTOR:
            skipped_sequences += 1
            continue
        
        # Debug first few sequences
        if idx < 5:
            print(f"\nSequence {idx}:")
            print(f"Original length: {len(seq)}")
        
        # Extract chunks
        for i in range(0, len(seq) - chunk_size * UPSCALE_FACTOR + 1, chunk_size * UPSCALE_FACTOR):
            # Extract target chunk (high resolution)
            target_chunk = seq[i:i + chunk_size * UPSCALE_FACTOR]
            
            # Create downsampled input chunk by averaging groups of UPSCALE_FACTOR samples
            input_chunk = np.mean(target_chunk.reshape(-1, UPSCALE_FACTOR), axis=1)
            # input_chunk = augment_chunk(input_chunk)  # Add augmentation
            
            # Verify chunk sizes
            if len(input_chunk) == chunk_size and len(target_chunk) == chunk_size * UPSCALE_FACTOR:
                # Reshape to (length, 1) for the model
                input_chunks.append(input_chunk.reshape(-1, 1))
                target_chunks.append(target_chunk.reshape(-1, 1))
    
    # Convert to numpy arrays if we have any chunks
    if input_chunks:
        input_chunks = np.array(input_chunks)
        target_chunks = np.array(target_chunks)
        
        print(f"\nChunking results:")
        print(f"Created {len(input_chunks)} chunks")
        print(f"Skipped {skipped_sequences} sequences")
        print(f"Input chunk shape: {input_chunks.shape}")
        print(f"Target chunk shape: {target_chunks.shape}")
        
        # Verify chunk statistics
        print(f"\nChunk statistics:")
        print(f"Input chunks - Min: {input_chunks.min():.4f}, Max: {input_chunks.max():.4f}")
        print(f"Target chunks - Min: {target_chunks.min():.4f}, Max: {target_chunks.max():.4f}")
    else:
        print("\nWarning: No valid chunks created!")
        print(f"Skipped {skipped_sequences} sequences")
    
    return input_chunks, target_chunks
    

# Process sequences into chunks
print("\nProcessing training sequences...")
X_train, y_train = process_sequences_to_chunks(X_train)
print("\nProcessing test sequences...")
X_test, y_test = process_sequences_to_chunks(X_test)

print("\nFinal shapes:")
print(f"Training - X (input): {X_train.shape}, y (target): {y_train.shape}")
print(f"Testing  - X (input): {X_test.shape}, y (target): {y_test.shape}")

# Verify upscaling relationship
if X_train.shape[1] * UPSCALE_FACTOR != y_train.shape[1]:
    raise ValueError(f"Training data upscale factor mismatch: {y_train.shape[1] / X_train.shape[1]:.1f}x")
if X_test.shape[1] * UPSCALE_FACTOR != y_test.shape[1]:
    raise ValueError(f"Test data upscale factor mismatch: {y_test.shape[1] / X_test.shape[1]:.1f}x")

# Validate that we have data before creating datasets
if len(X_train) == 0 or len(y_train) == 0:
    raise ValueError("No valid training chunks were created. Please check the data and chunking parameters.")

if len(X_test) == 0 or len(y_test) == 0:
    raise ValueError("No valid testing chunks were created. Please check the data and chunking parameters.")

# Create datasets and dataloaders
print("\nCreating datasets...")
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

print(f"Dataset sizes:")
print(f"Training: {len(train_dataset)} chunks")
print(f"Testing: {len(test_dataset)} chunks")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()

num_epochs = NUM_EPOCHS

print("\nTraining configuration:")
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of epochs: {num_epochs}")
print(f"Upscale factor: {UPSCALE_FACTOR}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Training chunks: {len(X_train)}")
print(f"Testing chunks: {len(X_test)}")

print("\nStarting training...")

for epoch in range(num_epochs):
    start_time = time.time()
    
    # Run one training epoch
    train_loss, train_snr = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    
    # Run one testing (validation) epoch
    test_loss, test_snr = test_epoch(model, test_loader, criterion, DEVICE)
    
    end_time = time.time()
    
    print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Test Loss: {test_loss:.4f} | "
          f"Time: {end_time - start_time:.2f}s")
    print(f"            | Train SNR: {train_snr:.2f} dB | Test SNR: {test_snr:.2f} dB")

print("Training finished.")

# summary(model)

# Save the model
torch.save(model.state_dict(), 'tfilm_superres_upscale_model.pth')