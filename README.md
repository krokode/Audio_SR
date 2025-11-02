# About
A PyTorch reproduction of the **Temporal FiLM** (Birnbaum, S., et al., 2019 [NeurIPS]) super-resolution [method](https://github.com/haoheliu/versatile_audio_super_resolution).

# Environment
## Setup
### Local
Clone the project from the desired parent directory on the local device:
```bash
git clone https://github.com/krokode/Audio_SR.git
cd Audio_SR
```

Setup the software environment:

#### Linux
```bash
. setup.sh
. activate.sh
```

#### Windows
```powershell
setup.ps1
activate.ps1
```

### VSC Supercomputer
Clone the project onto the data partition:
```bash
cd $VSC_DATA
git clone https://github.com/krokode/Audio_SR.git
cd Audio_SR
```

Setup the software environment:
```bash
. vsc_setup_wice.sh
. vsc_activate_wise.sh
```

# Datasets
## Download
### Local
```bash
cd data/vctk
python arc_load_unpack.py
```

### VSC Supercomputer
Upload [data]("http://www.udialogue.org/download/VCTK-Corpus.tar.gz") with WinSCP to the `$VSC_DATA/Audio_SR/data/vctk` partition.

Unpack the data acrhive:
```bash
cd $VSC_DATA/Audio_SR/data/vctk
tar -xvf VCTK-Corpus.tar.gz
```

## Prepare
Preprocess extracted data as h5 files

```bash
python prepare_dataset.py
```

# Train
Train model on h5 files for 150 epochs (update NUM_EPOCHS inside `run.py`)
```bash
cd ../../src
python run.py
```

# Visualize
It will create 4 times lower resolution example then pass it though model and create predicted wav file.

Pass any WAV file in high-resolution, for example p270_002.wav:
```bash
python visualize.py --model best_model_V1_6.pth --wav p270_002.wav --out output
```
Audio results and spectrograms will be available in `/visualizations/output/`.
