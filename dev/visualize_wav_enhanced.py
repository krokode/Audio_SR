"""Generate enhanced visualizations for audio super-resolution results.

This script creates comprehensive visualizations comparing low-res (LR),
high-res (HR), and predicted (PR) audio files, including:
- Waveform comparisons
- Magnitude spectrograms (linear and log frequency)
- Phase spectrograms
- Mel spectrograms

Usage:
    python dev/visualize_wav_enhanced.py --lr input_lr.wav --hr original.wav --pr predicted.wav --out figures/result1
"""
import argparse
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

def load_audio(path, sr=None):
    """Load audio file and return signal and sample rate."""
    y, sr = librosa.load(path, sr=sr)
    return y, sr

def plot_waveforms(hr, lr, pr, sr_hr, sr_lr, out_path, duration=2.0):
    """Plot waveform comparisons of the first N seconds."""
    # Take first N seconds
    n_samples_hr = int(duration * sr_hr)
    n_samples_lr = int(duration * sr_lr)
    
    hr = hr[:n_samples_hr]
    lr = lr[:n_samples_lr]
    pr = pr[:n_samples_hr]
    
    t_hr = np.linspace(0, duration, len(hr))
    t_lr = np.linspace(0, duration, len(lr))
    
    plt.figure(figsize=(15, 8))
    
    # High-res (original)
    plt.subplot(3, 1, 1)
    plt.plot(t_hr, hr, 'b-', label='Original (HR)', alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.title(f'Waveform Comparison (first {duration}s)')
    plt.ylabel('Amplitude')
    
    # Low-res
    plt.subplot(3, 1, 2)
    plt.plot(t_lr, lr, 'r-', label='Low-res (LR)', alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.ylabel('Amplitude')
    
    # Predicted
    plt.subplot(3, 1, 3)
    plt.plot(t_hr, pr, 'g-', label='Predicted (PR)', alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_magnitude_spectrogram(y, sr, out_path, title, y_axis='log', cmap='magma'):
    """Plot magnitude spectrogram with options."""
    plt.figure(figsize=(12, 6))
    
    D = librosa.stft(y)
    mag_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    img = librosa.display.specshow(
        mag_db,
        sr=sr,
        y_axis=y_axis,
        x_axis='time',
        cmap=cmap
    )
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(f'{title} - Magnitude Spectrogram')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_phase_spectrogram(y, sr, out_path, title, y_axis='log'):
    """Plot phase spectrogram."""
    plt.figure(figsize=(12, 6))
    
    D = librosa.stft(y)
    phase = np.angle(D)
    
    img = librosa.display.specshow(
        phase,
        sr=sr,
        y_axis=y_axis,
        x_axis='time',
        cmap='twilight'
    )
    plt.colorbar(img, label='Radians')
    plt.title(f'{title} - Phase Spectrogram')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_mel_spectrogram(y, sr, out_path, title, n_mels=128):
    """Plot mel-frequency spectrogram."""
    plt.figure(figsize=(12, 6))
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        y_axis='mel',
        x_axis='time',
        cmap='magma'
    )
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(f'{title} - Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_chromagram(y, sr, out_path, title):
    """Plot chromagram showing harmonic content."""
    plt.figure(figsize=(12, 6))
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    img = librosa.display.specshow(
        chroma,
        sr=sr,
        y_axis='chroma',
        x_axis='time',
        cmap='coolwarm'
    )
    plt.colorbar(img)
    plt.title(f'{title} - Chromagram')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def generate_visualizations(hr_path, lr_path, pr_path, out_prefix):
    """Generate all visualizations for a set of audio files."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading audio files...")
    hr, sr_hr = load_audio(hr_path)
    lr, sr_lr = load_audio(lr_path)
    pr, sr_pr = load_audio(pr_path)
    
    print("\nGenerating waveform comparison...")
    plot_waveforms(hr, lr, pr, sr_hr, sr_lr, 
                  out_prefix / 'waveforms.png')
    
    print("\nGenerating magnitude spectrograms...")
    # Linear frequency
    plot_magnitude_spectrogram(hr, sr_hr, out_prefix / 'hr_mag_linear.png',
                             'Original (HR)', y_axis='linear')
    plot_magnitude_spectrogram(lr, sr_lr, out_prefix / 'lr_mag_linear.png',
                             'Low-res (LR)', y_axis='linear')
    plot_magnitude_spectrogram(pr, sr_pr, out_prefix / 'pr_mag_linear.png',
                             'Predicted (PR)', y_axis='linear')
    
    # Log frequency
    plot_magnitude_spectrogram(hr, sr_hr, out_prefix / 'hr_mag_log.png',
                             'Original (HR)', y_axis='log')
    plot_magnitude_spectrogram(lr, sr_lr, out_prefix / 'lr_mag_log.png',
                             'Low-res (LR)', y_axis='log')
    plot_magnitude_spectrogram(pr, sr_pr, out_prefix / 'pr_mag_log.png',
                             'Predicted (PR)', y_axis='log')
    
    print("\nGenerating phase spectrograms...")
    plot_phase_spectrogram(hr, sr_hr, out_prefix / 'hr_phase.png',
                          'Original (HR)')
    plot_phase_spectrogram(pr, sr_pr, out_prefix / 'pr_phase.png',
                          'Predicted (PR)')
    
    print("\nGenerating mel spectrograms...")
    plot_mel_spectrogram(hr, sr_hr, out_prefix / 'hr_mel.png',
                        'Original (HR)')
    plot_mel_spectrogram(lr, sr_lr, out_prefix / 'lr_mel.png',
                        'Low-res (LR)')
    plot_mel_spectrogram(pr, sr_pr, out_prefix / 'pr_mel.png',
                        'Predicted (PR)')
    
    print("\nGenerating chromagrams...")
    plot_chromagram(hr, sr_hr, out_prefix / 'hr_chroma.png',
                   'Original (HR)')
    plot_chromagram(pr, sr_pr, out_prefix / 'pr_chroma.png',
                   'Predicted (PR)')
    
    print(f"\nAll visualizations saved to {out_prefix}/")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hr', required=True, help='High-resolution audio file (original)')
    parser.add_argument('--lr', required=True, help='Low-resolution audio file (input)')
    parser.add_argument('--pr', required=True, help='Predicted audio file (model output)')
    parser.add_argument('--out', required=True, help='Output directory prefix for visualizations')
    args = parser.parse_args()
    
    generate_visualizations(args.hr, args.lr, args.pr, args.out)

if __name__ == '__main__':
    main()