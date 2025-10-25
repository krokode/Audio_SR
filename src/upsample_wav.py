import os
import argparse
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from model_ds_v2 import create_tfilm_super_resolution
from utils import get_spectrum, save_spectrum

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def load_model(checkpoint_path, upscale_factor=4, quality_mode=True):
    model = create_tfilm_super_resolution(upscale_factor=upscale_factor, quality_mode=quality_mode)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def process_wav(model, wav_path, out_prefix, sr=16000, r=4, patch_size=8192):
    # Load high-res waveform
    x_hr, fs = librosa.load(wav_path, sr=sr)

    # Create low-res by decimating
    x_lr = librosa.resample(x_hr, orig_sr=fs, target_sr=fs//r)

    # Pad to multiple of patch_size
    pad_len = (patch_size - (len(x_hr) % patch_size)) % patch_size
    x_hr_padded = np.pad(x_hr, (0, pad_len), mode='constant')

    # Downsample padded to create model input
    x_lr_padded = librosa.resample(x_hr_padded, orig_sr=fs, target_sr=fs//r)

    # Prepare model input shape: (batch, channels, seq_len)
    x_input = torch.tensor(x_lr_padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Run model (may need to chop if very long)
    with torch.no_grad():
        y_pred = model(x_input)
    y_pred = y_pred.cpu().numpy().flatten()

    # Crop predicted to original length
    y_pred = y_pred[:len(x_hr_padded)]
    x_hr_padded = x_hr_padded[:len(y_pred)]
    x_lr = x_lr[:len(y_pred)//r]

    # Upsample x_lr to match y_pred's length for summing
    x_lr_upsampled = librosa.resample(x_lr, orig_sr=fs//r, target_sr=fs)
    x_lr_upsampled = x_lr_upsampled[:len(y_pred)]  # Ensure same length
    
    # Create summed versions (normalize to prevent clipping)
    # Sum 1: LR + PR (Enhancement added to input)
    y_sum_lr = y_pred + x_lr_upsampled
    y_sum_lr = y_sum_lr / np.max(np.abs(y_sum_lr))
    
    # Sum 2: HR + PR (Enhancement added to reference)
    y_sum_hr = y_pred + x_hr_padded[:len(y_pred)]
    y_sum_hr = y_sum_hr / np.max(np.abs(y_sum_hr))

    # Convert output prefix to Path object first
    out_prefix = Path(out_prefix)

    # Create visualization directory structure under `visualizations/<out_prefix>`
    visualization_dir = Path("visualizations")
    viz_target = visualization_dir / out_prefix
    os.makedirs(viz_target, exist_ok=True)

    # Base filename (used as prefix inside viz_target)
    base_name = out_prefix.name

    # Save all audio files into the visualization target directory
    sf.write(str(viz_target / (base_name + '.lr.wav')), x_lr, int(fs / r))
    sf.write(str(viz_target / (base_name + '.hr.wav')), x_hr_padded, fs)
    sf.write(str(viz_target / (base_name + '.pr.wav')), y_pred, fs)
    sf.write(str(viz_target / (base_name + '.sum_lr.wav')), y_sum_lr, fs)  # LR + PR
    sf.write(str(viz_target / (base_name + '.sum_hr.wav')), y_sum_hr, fs)  # HR + PR

    # Create comparison visualization (saved inside viz_target)
    signals = [x_hr_padded, x_lr_upsampled, y_pred, y_sum_lr, y_sum_hr]
    srs = [fs] * 5
    names = ['Original (HR)', 'Input (LR)', 'Predicted (PR)', 
             'Combined (LR+PR)', 'Combined (HR+PR)']
    plot_comparison(signals, srs, names, str(viz_target / (base_name + '.comparison.png')))

    # Save individual spectrograms inside viz_target
    S_pr = get_spectrum(y_pred)
    save_spectrum(S_pr, outfile=str(viz_target / (base_name + '.pr.png')))
    S_hr = get_spectrum(x_hr_padded)
    save_spectrum(S_hr, outfile=str(viz_target / (base_name + '.hr.png')))
    S_sum_lr = get_spectrum(y_sum_lr)
    save_spectrum(S_sum_lr, outfile=str(viz_target / (base_name + '.sum_lr.png')))
    S_sum_hr = get_spectrum(y_sum_hr)
    save_spectrum(S_sum_hr, outfile=str(viz_target / (base_name + '.sum_hr.png')))
    S_lr = get_spectrum(x_lr_upsampled)
    save_spectrum(S_lr, outfile=str(viz_target / (base_name + '.lr.png')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--wav', required=True, help='Input WAV file')
    parser.add_argument('--out', required=True, help='Output prefix for saved files')
    parser.add_argument('--sr', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--r', type=int, default=4, help='Downsampling ratio (upsample factor)')
    parser.add_argument('--patch_size', type=int, default=8192)
    args = parser.parse_args()

    model = load_model(args.model, upscale_factor=args.r, quality_mode=False)
    process_wav(model, args.wav, args.out, sr=args.sr, r=args.r, patch_size=args.patch_size)


if __name__ == '__main__':
    main()
