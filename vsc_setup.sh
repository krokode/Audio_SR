module load cluster/genius/gpu_p100 PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
pip install pip --upgrade

# pip install ... --no-cache-dir --no-build-isolation
