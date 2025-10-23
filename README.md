1. Clone REPO
```
git clone https://github.com/krokode/Audio_SR.git
cd AUDIO_SR
```
2. Create vertual environment and install required packages
```
pip install -r requirements.txt
```
3. Data for training load and extract
```
cd data/vctk
python3 arc_load_unpack.py
```
4. Preprocess extracted data
For Linux/MacOS
```
cd speaker1
makefile.sh
```
For Windows
```
makefile.ps1
```
5. Train model for 50 epochs
```
cd ../../../src
python3 run.py
```
6. To train for more epochs edit run.py NUM_EPOCHS variable

