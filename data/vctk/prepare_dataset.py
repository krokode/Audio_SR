from pathlib import Path
import argparse
import os
import subprocess
import sys
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sampling_rate', type=int, default=16000, help='audio sampling rate')
    parser.add_argument('--scale', type=int, default=2, help='scaling factor')
    parser.add_argument('--window_size', type=int, default=8192, help='receptive filed size (patches). -1 for no patching')
    parser.add_argument('--window_stride', type=int, default=4096, help='overlap size between consecutive patches')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size used by AI for complete minibatches')
    parser.add_argument('--interpolate', action='store_true', help='interpolate low-res patches with cubic splines')
    parser.add_argument('--low_pass', action='store_true', help='apply low-pass filter when generating low-res patches for antialiasing')
    parser.add_argument('--out_dir', help='folder for current preprocessing config')

    args = parser.parse_args()

    # Folder to store wav lists txt
    datasets_dir = Path(args.out_dir,
                        f"{args.sampling_rate}" \
                        f"_x{args.scale}" \
                        f"_w{args.window_size}" \
                        f"_s{args.window_stride}" \
                        f"_{args.batch_size}" \
                        f"_{'' if args.low_pass else 'no'}lp" \
                        f"_{'' if args.interpolate else 'no'}interpolate")

    os.makedirs(datasets_dir, exist_ok=True)

    vctk_dir = Path('VCTK-Corpus', 'wav48')

    # List subfolders in vctk_dir
    speaker_dirs = sorted([d for d in vctk_dir.iterdir() if d.is_dir()])

    for speaker_dir in tqdm(speaker_dirs):
        wavs = speaker_dir.name
        wav_path = Path(vctk_dir, wavs)
        
        # List wav files in subfolders
        wavs_in_folder = list(speaker_dir.glob('*.wav'))
        split_point = int(len(wavs_in_folder) * 0.7)
        
        # Create train_files list 70%     
        train_files = wavs_in_folder[:split_point]
        # Create train_files list 30%
        val_files = wavs_in_folder[split_point:]

        train_files_txt = Path(datasets_dir, f'{wavs}_train_files.txt')
        with open(train_files_txt, 'w') as f:
            for i in train_files:
                f.write(f'{str(i.name)}\n')

        val_files_txt = Path(datasets_dir, f'{wavs}_val_files.txt')
        with open(val_files_txt, 'w') as f:
            for i in val_files:
                f.write(f'{str(i.name)}\n')

        train_dataset_name = f'{wavs}_train.h5'
        val_dataset_name = f'{wavs}_val.h5'
        train_dataset = Path(datasets_dir, train_dataset_name)
        val_dataset = Path(datasets_dir, val_dataset_name)

        script = 'prep_vctk.py'
        if not Path(script).exists():
            print(f"Error: {script} not found in current directory")
            sys.exit(1)

        params_train = {
            '--file_list' : train_files_txt,
            '--in_dir' : wav_path,
            '--out' : train_dataset,
            '--scale' : args.scale,
            '--sr' : args.sampling_rate,
            '--dimension' : args.window_size,
            '--stride' : args.window_stride,
            '--batch_size' : args.batch_size,
            '--interpolate' : args.interpolate,
            '--low_pass' : args.low_pass,
        }

        args_train = []
        for key, value in params_train.items():
            if isinstance(value, bool):
                if value:
                    args_train.append(key)
            else:
                args_train.extend([key, str(value)])

        params_val = {
            '--file_list' : val_files_txt,
            '--in_dir' : wav_path,
            '--out' : val_dataset,
            '--scale' : args.scale,
            '--sr' : args.sampling_rate,
            '--dimension' : args.window_size,
            '--stride' : args.window_stride,
            '--batch_size' : args.batch_size,
            '--interpolate' : args.interpolate,
            '--low_pass' : args.low_pass,
        }

        args_val = []
        for key, value in params_val.items():
            if isinstance(value, bool):
                if value:
                    args_val.append(key)
            else:
                args_val.extend([key, str(value)])

        try:
            result_train = subprocess.run(
                [sys.executable, script] + args_train,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed args_train: {e}")
        
        try:
            result_val = subprocess.run(
                [sys.executable, script] + args_val,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed args_val: {e}")
