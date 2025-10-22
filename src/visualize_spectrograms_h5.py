"""Generate spectrogram visualizations from HDF5 training/test datasets used by `src/run.py`.

Saves examples to `src/figures_h5/` as PNGs.

Usage:
    python src/visualize_spectrograms_h5.py
"""
import os
import numpy as np
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
from utils import load_h5

ROOT = Path(__file__).parent
OUT_DIR = ROOT / 'figures_h5'
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
N_EXAMPLES = 4


def ensure_1d(y):
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def save_spectrogram(y, sr, out_path, title=None, n_fft=1024, hop_length=256):
    y = ensure_1d(y)
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


def save_mel(y, sr, out_path, title=None, n_fft=1024, hop_length=256, n_mels=128):
    y = ensure_1d(y)
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
    root_dir = Path(__file__).parent.parent
    train_file_path = root_dir / 'data' / 'vctk' / 'speaker1' / 'vctk-speaker1-train.4.16000.8192.4096.h5'
    test_file_path = root_dir / 'data' / 'vctk' / 'speaker1' / 'vctk-speaker1-val.4.16000.8192.4096.h5.tmp'

    print(f"Loading HDF5 train/test from:\n  {train_file_path}\n  {test_file_path}")
    X_train, y_train = load_h5(train_file_path)
    X_test, y_test = load_h5(test_file_path)

    examples = min(N_EXAMPLES, X_train.shape[0])
    for i in range(examples):
        x = X_train[i]
        y = y_train[i]

        # Normalize for visualization only
        def norm(a):
            a = ensure_1d(a)
            m = np.abs(a).max()
            return a / m if m > 0 else a

        x_vis = norm(x)
        y_vis = norm(y)

        save_spectrogram(x_vis, SAMPLE_RATE, OUT_DIR / f'ex_{i}_input_lin.png', title=f'ex_{i} input linear')
        save_mel(x_vis, SAMPLE_RATE, OUT_DIR / f'ex_{i}_input_mel.png', title=f'ex_{i} input mel')
        save_spectrogram(y_vis, SAMPLE_RATE, OUT_DIR / f'ex_{i}_target_lin.png', title=f'ex_{i} target linear')
        save_mel(y_vis, SAMPLE_RATE, OUT_DIR / f'ex_{i}_target_mel.png', title=f'ex_{i} target mel')

        print(f"Saved example {i} to {OUT_DIR}")


if __name__ == '__main__':
    main()
