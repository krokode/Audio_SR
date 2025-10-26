import os
import argparse
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torchinfo import summary
from model_ds_v1_6 import create_tfilm_super_resolution
from utils import get_spectrum, save_spectrum

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NORMALIZE_INPUT = True  # Whether to normalize input audio to [-1, 1]
QUALITY_MODE = True  # Whether the model is in quality enhancement mode (output same length as input)
UPSCALE_FACTOR = 4

# Run inference on a WAV file using a trained TFiLM model.
# Usage (Linux/Mac):
#     python src/upsample_wav.py --model path/to/model.pth --wav input.wav --out output
# Usage (PowerShell):
#     python .\src\upsample_wav.py --model checkpoint.pth --wav input.wav --out out_prefix

# This script:
# - Loads the model architecture from `src/model_ds_v2.py`
# - Loads weights from a checkpoint
# - Loads `input.wav` (resamples if necessary)
# - Produces a low-res version (decimated) and saves it as `<out> .lr.wav`
# - Runs the model to generate a prediction and saves `<out>.pr.wav` and `<out>.hr.wav`
# - Also saves spectrogram PNGs alongside the audio files

def plot_comparison(signals, srs, names, out_path, duration=3.0):
    """Plot waveform and spectrogram comparisons.
    
    Args:
        signals: List of signals to compare
        srs: List of sample rates for each signal
        names: List of names for each signal
        out_path: Output path for plot
        duration: Duration in seconds to plot
    """
    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 2, figsize=(15, 4*n_signals))
    
    for i, (signal, sr, name) in enumerate(zip(signals, srs, names)):
        # Take first N seconds
        n_samples = int(duration * sr)
        signal = signal[:n_samples]
        t = np.linspace(0, duration, len(signal))
        
        # Waveform
        axes[i,0].plot(t, signal, label=name)
        axes[i,0].set_title(f'{name} - Waveform')
        axes[i,0].set_xlabel('Time (s)')
        axes[i,0].set_ylabel('Amplitude')
        axes[i,0].grid(True)
        
        # Spectrogram
        D = librosa.stft(signal)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log',
                                     ax=axes[i,1])
        axes[i,1].set_title(f'{name} - Spectrogram')
        plt.colorbar(img, ax=axes[i,1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_model(checkpoint_path, upscale_factor=6, quality_mode=False):
    """Load trained model from checkpoint."""
    # Instanciate model as it was during training
    model = create_tfilm_super_resolution(
        upscale_factor=upscale_factor,
        quality_mode=quality_mode,
        base_channels=64,
        tfilm_hidden_size=128,
        block_size=256
    )
    
    # Load checkpoint file
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Normalize to a state_dict mapping
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        print("Checkpoint contains 'model_state_dict' key; using that for loading.")
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        print("Checkpoint does not contain 'model_state_dict' key; using raw checkpoint.")
    else:
        raise TypeError(f"Unexpected checkpoint object type: {type(ckpt)}")

    # Try strict load first
    try:
        model.load_state_dict(state_dict)
        print("Checkpoint loaded (strict).")
        print("Model Loaded all parameters successfully.")
        summary(model)        
    except Exception as exc:
        # Strict load failed - attempt a filtered partial load
        print("Strict load failed:", exc)
        print("Attempting partial load of matching parameters...")

        model_sd = model.state_dict()
        filtered = {}
        skipped = []

        for k, v in state_dict.items():
            if k in model_sd:
                if tuple(v.shape) == tuple(model_sd[k].shape):
                    filtered[k] = v
                else:
                    skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            else:
                skipped.append((k, 'unexpected_in_checkpoint', None))

        # Load filtered params non-strictly
        model.load_state_dict(filtered, strict=False)

        print("Partial checkpoint load complete.")
        summary(model)

        print(f"Partially loaded {len(filtered)} parameters; skipped {len(skipped)} entries.")
        if len(skipped) > 0:
            print("Skipped examples (name, ckpt_shape, model_shape):")
            for s in skipped[:16]:
                print(" ", s)

    model.to(DEVICE)
    model.eval()
    return model

def normalize(x):
    """Normalize audio to [-1, 1] range."""
    x = np.asarray(x, dtype=np.float32)
    max_val = np.abs(x).max()
    if max_val > 0:
        return x / max_val
    return x

def process_wav(model, wav_path, out_prefix, sr=16000, r=UPSCALE_FACTOR, patch_size=8192, normalize_input=True):
    if normalize_input:
        print("Normalizing hr input audio...")
        x_hr, fs = librosa.load(wav_path, sr=sr)
        x_hr = normalize(x_hr)
    else:
        # Load high-res waveform
        x_hr, fs = librosa.load(wav_path, sr=sr)

    # This mimics the training data (blurry_N, sharp_N)
    fs_lr_target = fs // r
    print(f"Generating LR input by downsampling to {fs_lr_target}Hz and upsampling back to {fs}Hz...")

    # Downsample x_hr (decimate) to get lr
    x_lr_down = librosa.resample(x_hr, orig_sr=fs, target_sr=fs_lr_target)

    # Upsample (interpolate) back to original sample rate
    # This creates the "blurry" input for the quality enhancement model
    x_lr = librosa.resample(x_lr_down, orig_sr=fs_lr_target, target_sr=fs)

    # Ensure generated LR audio has same sample rate and approx length
    fs_lr = fs 
    
    # Pad/trim x_lr to match x_hr length exactly
    hr_len = len(x_hr)
    lr_len = len(x_lr)
    if hr_len > lr_len:
        x_lr = np.pad(x_lr, (0, hr_len - lr_len))
    elif lr_len > hr_len:
        x_lr = x_lr[:hr_len]

    # Store original length for final cropping
    original_length = len(x_hr)
    print(f"Original audio length: {original_length} samples")    

    # Prepare low-res for model input
    x_input = torch.tensor(x_lr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Run model
    print("Running model inference...")
    with torch.no_grad():
        y_pred = model(x_input)
    
    # Crop and flatten predicted audio
    # In Quality Mode, model output length should match input length
    y_pred = y_pred.squeeze().cpu().numpy()
    
    # Final check to ensure exact length match with HR target
    if len(y_pred) > original_length:
        y_pred = y_pred[:original_length]
    elif len(y_pred) < original_length:
        y_pred = np.pad(y_pred, (0, original_length - len(y_pred)))

    # Convert output prefix to Path object first
    out_prefix = Path(out_prefix)

    # Create visualization directory structure under `visualizations/<out_prefix>`
    visualization_dir = Path("visualizations")
    viz_target = visualization_dir / out_prefix
    os.makedirs(viz_target, exist_ok=True)

    # Base filename (used as prefix inside viz_target)
    base_name = out_prefix.name

    # Save all audio files into the visualization target directory
    sf.write(str(viz_target / (base_name + '.lr.wav')), x_lr, fs_lr)
    sf.write(str(viz_target / (base_name + '.hr.wav')), x_hr, fs)
    sf.write(str(viz_target / (base_name + '.pr.wav')), y_pred, fs)
   
    # Create comparison visualization (saved inside viz_target)
    signals = [x_hr, x_lr, y_pred]
    srs = [fs] * 3
    names = ['Original (HR)', 'Input (LR)', 'Predicted (PR)']
    plot_comparison(signals, srs, names, str(viz_target / (base_name + '.comparison.png')))

    # Save individual spectrograms inside viz_target
    S_pr = get_spectrum(y_pred)
    save_spectrum(S_pr, outfile=str(viz_target / (base_name + '.pr.png')))
    S_hr = get_spectrum(x_hr)
    save_spectrum(S_hr, outfile=str(viz_target / (base_name + '.hr.png')))
    S_lr = get_spectrum(x_lr)
    save_spectrum(S_lr, outfile=str(viz_target / (base_name + '.lr.png')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--wav', required=True, help='Input HR WAV file')
    parser.add_argument('--out', required=True, help='Output prefix for saved files')
    parser.add_argument('--sr', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--r', type=int, default=UPSCALE_FACTOR, help='Downsampling ratio (upsample factor)')
    parser.add_argument('--patch_size', type=int, default=8192)
    args = parser.parse_args()

    model = load_model(args.model, upscale_factor=args.r, quality_mode=QUALITY_MODE)  # Match training configuration
    process_wav(model, args.wav, args.out, sr=args.sr, r=args.r, patch_size=args.patch_size, normalize_input=NORMALIZE_INPUT)


if __name__ == '__main__':
    main()
