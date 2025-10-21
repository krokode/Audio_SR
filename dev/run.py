import os
from .utils import load_h5, upsample_wav

train_file_path = os.path.join('data', 'train_data.h5') # Path to training data
test_file_path = os.path.join('data', 'test_data.h5')   # Path to test data

X_train, y_train = load_h5(train_file_path)
X_test, y_test = load_h5(test_file_path)