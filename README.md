1. Clone REPO
```
git clone https://github.com/krokode/Audio_SR.git
cd AUDIO_SR
```

2. Create virtual environment and install required packages
```
pip install -r requirements.txt
```

3. Data for training load and extract
```
cd data/vctk
python3 arc_load_unpack.py
```

4. Preprocess extracted data it will create dataset files h5 in /speaker1
Source wav files taken only from ./data/vctk/VCTK-Corpus/wav48/p225 for 
faster train experiment.
Can use folders from p226 to p376 for model improvements

For Linux/MacOS
```
cd speaker1
prepare_h5_train_test.sh
```
For Windows
```
prepare_h5_train_test.ps1
```
Can use folders from p226 to p376 for model improvements
```
cd ..
prepare_dataset.py
```

5. Train model on H5 files for 150 epochs best model to be saved
To train on small dataset
```
cd ../../../src
python3 run_v1_6.py
```
To train for more epochs edit run_v1_6.py NUM_EPOCHS variable
To train on multiple datasets
```
python3 run_v1_6_1.py
```

6. For predictions take any wav file in hi resolution for example p270_002.wav
```
python3 upsample_wav_v1_6.py --model best_model_V1_6.pth --wav p270_002.wav --out output_test1
``` 
It will create 4 times lower resolution example then pass it though model and create predicted wav file.
Everything will be saved in /visualizations/output_test1/
Also spectrogramms to be created for comparison 

