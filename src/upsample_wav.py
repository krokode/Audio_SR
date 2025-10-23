"""Run inference on a WAV file using a trained TFiLM model.

Usage (PowerShell):
    python .\src\upsample_wav.py --model checkpoint.pth --wav input.wav --out out_prefix

This script:
- Loads the model architecture from `src/model_ds_v1.py`
- Loads weights from a checkpoint
- Loads `input.wav` (resamples if necessary)
- Produces a low-res version (decimated) and saves it as `<out> .lr.wav`
- Runs the model to generate a prediction and saves `<out>.pr.wav` and `<out>.hr.wav`
- Also saves spectrogram PNGs alongside the audio files
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa
from model_ds_v2 import create_tfilm_super_resolution
from utils import get_spectrum, save_spectrum

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(checkpoint_path, upscale_factor=4, quality_mode=False):
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

    out_prefix = Path(out_prefix)
    sf.write(str(out_prefix) + '.lr.wav', x_lr, int(fs / r))
    sf.write(str(out_prefix) + '.hr.wav', x_hr_padded, fs)
    sf.write(str(out_prefix) + '.pr.wav', y_pred, fs)

    # Save spectrograms
    S_pr = get_spectrum(y_pred)
    save_spectrum(S_pr, outfile=str(out_prefix) + '.pr.png')
    S_hr = get_spectrum(x_hr_padded)
    save_spectrum(S_hr, outfile=str(out_prefix) + '.hr.png')
    S_lr = get_spectrum(x_lr)
    save_spectrum(S_lr, outfile=str(out_prefix) + '.lr.png')


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
