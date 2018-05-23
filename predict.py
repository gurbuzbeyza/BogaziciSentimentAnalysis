'''This script runs trained model on the specified dataset'''

import time
import pickle
from sklearn.linear_model import LogisticRegressionCV
from preprocess import tokenize_and_stem
from utils import TEST_SET_PATH, ML_MODEL_PATH, WORD2VEC_MODEL_FILE_PATH, TEST_SET_OUTPUT_PATH
from create_dataset import get_word_vectors, create_average_vectors, create_dictionary, calc_positivity, calc_negativity
import numpy as np
from random import shuffle

def read_file(file_path, all_tokens):
    '''Reads file with the specified file_name and returns preprocessed data'''
    f = open(file_path, 'r')
    dataset = [tokenize_and_stem(s, all_tokens) for s in f]
    return dataset

def get_preprocessed_data(file_path):
    all_tokens = set()
    all_data = read_file(file_path, all_tokens)
    return all_tokens, all_data


def get_dataset(all_tokens, all_data):
    word_vectors = get_word_vectors(WORD2VEC_MODEL_FILE_PATH)
    vect_dict = create_dictionary(all_tokens, word_vectors)
    data = []
    for d in all_data:
        avg_vec = create_average_vectors(d, vect_dict)
        if avg_vec.size != 0:
            avg_vec = np.append(avg_vec, calc_positivity(d))
            avg_vec = np.append(avg_vec, calc_negativity(d))
            data.append(avg_vec)
    shuffle(data)
    return data

def test(test_set_path):
    all_tokens, all_data = get_preprocessed_data(TEST_SET_PATH)
    with open(ML_MODEL_PATH, 'rb') as f_model:
        X = get_dataset(all_tokens, all_data)
        clf = pickle.load(f_model)
        result = clf.predict(X)
        return result

def main():
    with open(TEST_SET_OUTPUT_PATH, 'w') as f_write:
        result = test(TEST_SET_PATH)
        for r in result:
            f_write.write(str(r) + "\n")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- {} seconds ---".format(end - start))