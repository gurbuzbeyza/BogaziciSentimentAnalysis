'''This script runs trained model on the specified dataset'''

import time
import pickle
from sklearn.linear_model import LogisticRegressionCV
from preprocess import get_preprocessed_data
from utils import TEST_SET_PATH, ML_MODEL_PATH
from create_dataset import get_dataset

def test(test_set_path):
    all_tokens, all_data = get_preprocessed_data([TEST_SET_PATH])
    with open(ML_MODEL_PATH, 'rb') as f_model:
        (X, y) = get_dataset(all_tokens, all_data)
        clf = pickle.load(f_model)
        result = clf.score(X,y)
        return result

def main():
    print(test(TEST_SET_PATH))

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- {} seconds ---".format(end - start))