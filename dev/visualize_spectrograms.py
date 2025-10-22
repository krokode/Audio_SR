"""Generate and save spectrogram visualizations for sample inputs/targets.

Usage:
    python dev/visualize_spectrograms.py

This script uses the project's `load_full_files()` helper to load X/y and will
save a few example spectrograms (linear STFT magnitude and mel-spectrogram).

If librosa or matplotlib are missing, the script will instruct how to install them.
"""
import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from utils import load_full_files

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

N_EXAMPLES = 3
SAMPLE_RATE = 16000  # assumed dataset sample rate; adjust if different


def save_spectrogram(y, sr, out_path, title=None, n_fft=1024, hop_length=256):
    # Ensure 1D float array
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    # Use keyword args to avoid positional-argument signature issues across librosa versions
    D = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_mel_spectrogram(y, sr, out_path, title=None, n_fft=1024, hop_length=256, n_mels=128):
    # Ensure 1D float array
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    # Call with keyword args to be compatible with different librosa signatures
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    print("Loading data using load_full_files()...")
    X_train, y_train, X_test, y_test = load_full_files()

    def to_audio_array(x):
        # x may be list, numpy array, or nested; convert to 1D float array
        x = np.array(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.reshape(-1)
        return x

    examples = min(N_EXAMPLES, len(X_train))
    for i in range(examples):
        x = to_audio_array(X_train[i])
        y = to_audio_array(y_train[i])

        # If values are large (int16 range), normalize for visualization
        def norm_for_vis(a):
            a = a.astype(np.float32)
            m = np.abs(a).max()
            if m > 0:
                return a / m
            return a

        x_vis = norm_for_vis(x)
        y_vis = norm_for_vis(y)

        lin_out = os.path.join(OUT_DIR, f"example_{i}_input_linear.png")
        mel_out = os.path.join(OUT_DIR, f"example_{i}_input_mel.png")
        save_spectrogram(x_vis, SAMPLE_RATE, lin_out, title=f"Example {i} - Input (linear)")
        save_mel_spectrogram(x_vis, SAMPLE_RATE, mel_out, title=f"Example {i} - Input (mel)")

        lin_out_t = os.path.join(OUT_DIR, f"example_{i}_target_linear.png")
        mel_out_t = os.path.join(OUT_DIR, f"example_{i}_target_mel.png")
        save_spectrogram(y_vis, SAMPLE_RATE, lin_out_t, title=f"Example {i} - Target (linear)")
        save_mel_spectrogram(y_vis, SAMPLE_RATE, mel_out_t, title=f"Example {i} - Target (mel)")

        print(f"Saved example {i} spectrograms to {OUT_DIR}")


if __name__ == '__main__':
    main()
