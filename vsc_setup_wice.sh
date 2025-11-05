#! /bin/sh
module load cluster/wice/gpu PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 h5py/3.7.0-foss-2022a matplotlib/3.5.2-foss-2022a
python3 -m venv .venv_wice --system-site-packages
. .venv_wice/bin/activate
pip install pip --upgrade

# pip install ... --no-cache-dir --no-build-isolation
