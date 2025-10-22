"""Process long WAV files using chunked sliding-window inference.

Uses the same chunking strategy as training (from run_chunked_v2.py) but with
overlapping windows and crossfade to avoid boundary artifacts.

Usage:
    python dev/upsample_wav_chunked.py --model path/to/model.pth --wav input.wav --out output
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa
from scipy.signal import decimate
from model_ds_v1 import create_tfilm_super_resolution
from utils import get_spectrum, save_spectrum

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHUNK_SIZE = 8192  # Same as training
OVERLAP = 1024     # Overlap between chunks
XFADE_LEN = 512   # Crossfade length (must be <= OVERLAP)
UPSCALE_FACTOR = 4


def load_model(checkpoint_path):
    """Load trained model from checkpoint."""
    model = create_tfilm_super_resolution(
        upscale_factor=UPSCALE_FACTOR,
        quality_mode=False,
        base_channels=64,
        tfilm_hidden_size=128,
        block_size=256
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def normalize_chunk(x):
    """Normalize audio chunk to [-1, 1] range."""
    x = np.asarray(x, dtype=np.float32)
    max_val = np.abs(x).max()
    if max_val > 0:
        return x / max_val
    return x


def process_chunk(model, chunk, normalize=True):
    """Process a single audio chunk through the model."""
    if normalize:
        chunk = normalize_chunk(chunk)
    
    # Add batch and channel dims: (samples,) -> (1, 1, samples)
    x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        y = model(x)
    
    return y.cpu().numpy().squeeze()


def crossfade(chunk1, chunk2, xfade_len):
    """Crossfade two chunks at their overlap."""
    if xfade_len == 0:
        return chunk1
    
    # Create fade curves
    fade_out = np.linspace(1, 0, xfade_len)
    fade_in = np.linspace(0, 1, xfade_len)
    
    # Apply crossfade
    chunk1[-xfade_len:] *= fade_out
    chunk2[:xfade_len] *= fade_in
    
    # Combine
    chunk1[-xfade_len:] += chunk2[:xfade_len]
    return chunk1


def process_wav_chunked(model, wav_path, out_prefix, sr=16000):
    """Process a WAV file using overlapping chunks with crossfade."""
    # Load audio
    x_hr, fs = librosa.load(wav_path, sr=sr)
    print(f"Loaded audio: {len(x_hr)} samples @ {fs}Hz")
    
    # Create low-res version for processing
    x_lr = librosa.resample(x_hr, orig_sr=fs, target_sr=fs//UPSCALE_FACTOR)
    print(f"Downsampled to {len(x_lr)} samples @ {fs//UPSCALE_FACTOR}Hz")
    
    # Pad if needed
    if len(x_lr) % CHUNK_SIZE != 0:
        pad_len = CHUNK_SIZE - (len(x_lr) % CHUNK_SIZE)
        x_lr = np.pad(x_lr, (0, pad_len), mode='constant')
    
    # Process in overlapping chunks
    chunk_size = CHUNK_SIZE
    hop_size = chunk_size - OVERLAP
    num_chunks = (len(x_lr) - chunk_size) // hop_size + 1
    
    # Calculate output size accounting for upscaling
    output_chunk_size = chunk_size * UPSCALE_FACTOR
    output_hop_size = hop_size * UPSCALE_FACTOR
    output_overlap = OVERLAP * UPSCALE_FACTOR
    output_xfade = XFADE_LEN * UPSCALE_FACTOR
    
    print(f"Processing {num_chunks} chunks with {OVERLAP} sample overlap...")
    output = np.zeros(len(x_lr) * UPSCALE_FACTOR)
    
    for i in range(num_chunks):
        # Extract chunk
        start = i * hop_size
        end = start + chunk_size
        chunk = x_lr[start:end]
        
        # Process through model
        pred = process_chunk(model, chunk)
        
        # Calculate output region accounting for upscaling
        out_start = i * output_hop_size
        out_end = out_start + output_chunk_size
        
        # Apply crossfade and add to output
        if i > 0:  # Not first chunk
            # Create views of the overlapping regions
            prev_region = output[out_start:out_start + output_xfade]
            curr_region = pred[:output_xfade]
            
            # Apply crossfade
            fade_out = np.linspace(1, 0, output_xfade)
            fade_in = np.linspace(0, 1, output_xfade)
            
            # Combine with crossfade
            output[out_start:out_start + output_xfade] = \
                prev_region * fade_out + curr_region * fade_in
            
            # Copy remaining part of prediction
            output[out_start + output_xfade:out_end] = pred[output_xfade:]
        else:  # First chunk
            output[out_start:out_end] = pred
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{num_chunks} chunks...")
    
    # Trim to original length * upscale
    final_len = len(x_hr)
    output = output[:final_len]
    x_lr = x_lr[:(final_len//UPSCALE_FACTOR)]
    
    # Save results
    out_prefix = Path(out_prefix)
    print(f"\nSaving outputs to {out_prefix}.*")
    
    sf.write(f"{out_prefix}.lr.wav", x_lr, fs//UPSCALE_FACTOR)
    sf.write(f"{out_prefix}.hr.wav", x_hr, fs)
    sf.write(f"{out_prefix}.pr.wav", output, fs)
    
    # Save spectrograms
    S_pr = get_spectrum(output)
    save_spectrum(S_pr, outfile=f"{out_prefix}.pr.png")
    S_hr = get_spectrum(x_hr)
    save_spectrum(S_hr, outfile=f"{out_prefix}.hr.png")
    S_lr = get_spectrum(x_lr)
    save_spectrum(S_lr, outfile=f"{out_prefix}.lr.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--wav', required=True, help='Input WAV file')
    parser.add_argument('--out', required=True, help='Output prefix for saved files')
    parser.add_argument('--sr', type=int, default=16000, help='Target sample rate')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    process_wav_chunked(model, args.wav, args.out, sr=args.sr)


if __name__ == '__main__':
    main()