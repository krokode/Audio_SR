from pathlib import Path
from utils import load_full

root_dir = Path(__file__).parent.parent  # Get project root directory
xtrain_file_path = root_dir / 'data' / 'vctk' / 'speaker1' / 'full-data-vctk-speaker1-train.4.16000.-1'      # Path to training data
ytrain_file_path = root_dir / 'data' / 'vctk' / 'speaker1' / 'full-label-vctk-speaker1-train.4.16000.-1'     # Path to test data
xtest_file_path = root_dir / 'data' / 'vctk' / 'speaker1' / 'full-data-vctk-speaker1-val.4.16000.-1.4096'       # Path to validation data
ytest_file_path = root_dir / 'data' / 'vctk' / 'speaker1' / 'full-label-vctk-speaker1-val.4.16000.-1.4096'      # Path to validation labels

X_train = load_full(xtrain_file_path)
y_train = load_full(ytrain_file_path)
X_test = load_full(xtest_file_path)
y_test = load_full(ytest_file_path)
)
print(X_train.keys())