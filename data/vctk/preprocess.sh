#! /bin/sh
python prepare_dataset.py --sampling_rate 16000 --scale 4 --window_size 8192 --window_stride 4096 --batch_size 128 --interpolate --low_pass --out_dir 'datasets/16kHz_x4_8192_4096_128_lp_interpolate'
