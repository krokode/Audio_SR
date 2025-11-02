from pathlib import Path
import os
import subprocess
import sys

# Folder to store datasets h5 format
datasets_Path = Path('datasets')
os.makedirs(datasets_Path, exist_ok=True)
# Folder to store wav lists txt
datasets_Path_lists_txt = Path(datasets_Path, 'files_txt')
os.makedirs(datasets_Path_lists_txt, exist_ok=True)

vctk_Path = Path('VCTK-Corpus','wav48')

# List subfolders in vctk_Path
speaker_dirs = sorted([d for d in vctk_Path.iterdir() if d.is_dir()])
# speaker_dirs = speaker_dirs[:5] # temporary split for testing

for speaker_dir in speaker_dirs:
    wavs = speaker_dir.name
    wav_Path = Path(vctk_Path, wavs)
    
    # List wav files in subfolders
    wavs_in_folder = list(speaker_dir.glob('*.wav'))
    split_point = int(len(wavs_in_folder) * 0.7)
    
    # Create train_files list 70%     
    train_files = wavs_in_folder[:split_point]
    # Create train_files list 30%
    val_files = wavs_in_folder[split_point:]

    train_files_txt = Path(datasets_Path_lists_txt, f'{wavs}_speaker1-train-files.txt')
    with open(train_files_txt, 'w') as f:
        for i in train_files:
            f.write(f'{str(i.name)}\n')

    val_files_txt = Path(datasets_Path_lists_txt, f'{wavs}_speaker1-val-files.txt')
    with open(val_files_txt, 'w') as f:
        for i in val_files:
            f.write(f'{str(i.name)}\n')

    train_dataset_name = f'{wavs}vctk-speaker1-train.4.16000.8192.4096.h5'
    val_dataset_name = f'{wavs}vctk-speaker1-val.4.16000.8192.4096.h5.tmp'
    train_dataset = Path(datasets_Path, train_dataset_name)
    val_dataset = Path(datasets_Path, val_dataset_name)

    script = 'prep_vctk.py'
    if not Path(script).exists():
        print(f"Error: {script} not found in current directory")
        sys.exit(1)

    params_train = {
        '--file-list' : str(train_files_txt),
        '--in-dir' : str(wav_Path),
        '--out' : str(train_dataset),
        '--scale' : '4',
        '--sr' : '16000',
        '--dimension' : '8192',
        '--stride' : '4096',
        '--interpolate' : 'store_true',
        '--low-pass' : 'store_true',
    }

    args_train = []
    for key, value in params_train.items():
        if value == 'store_true':
            args_train.append(key)
        else:
            args_train.extend([key, value])

    params_val = {
        '--file-list' : str(val_files_txt),
        '--in-dir' : str(wav_Path),
        '--out' : str(val_dataset),
        '--scale' : '4',
        '--sr' : '16000',
        '--dimension' : '8192',
        '--stride' : '4096',
        '--interpolate' : 'store_true',
        '--low-pass' : 'store_true',
    }

    args_val = []
    for key, value in params_val.items():
        if value == 'store_true':
            args_val.append(key)
        else:
            args_val.extend([key, value])

    try:
        result_train = subprocess.run(
            [sys.executable, script] + args_train,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Success! args_train_{wavs}")
        print(result_train.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed args_train: {e}")
    
    try:
        result_val = subprocess.run(
            [sys.executable, script] + args_val,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Success! args_val_{wavs}")
        print(result_val.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed args_val: {e}")