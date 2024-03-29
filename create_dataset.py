'''
This script creates the dataset from given training files.
It makes use of the pre-trained word2vec model.
'''
import sys
import time
import numpy as np
import pickle
from TurkishStemmer import TurkishStemmer
from gensim.models import Word2Vec
from random import shuffle
from utils import PREPROCESSED_TRAINING_FILE_PATH, WORD2VEC_MODEL_FILE_PATH, DATASET_PATH, POSITIVE_WORDS_PATH, NEGATIVE_WORDS_PATH


def prep(word_list):
    '''Returns preprocessed word_list'''
    stemmer = TurkishStemmer()
    word_list = [stemmer.stem(x) for x in word_list]
    word_list = [x.replace('ğ','g').replace('ı','i').replace('ç','c').replace('ş','s').replace('ü','u').replace('ö','o') for x in word_list]
    return word_list


def get_word_vectors(model_file_path):
    '''Returns previously trained word vectors'''
    model = Word2Vec.load(model_file_path)
    return model.wv

# Load previously trained word vectors
word_vectors = get_word_vectors(WORD2VEC_MODEL_FILE_PATH)

def get_similar_words():
    '''Returns previously trained word vectors'''
    global word_vectors

    p = open(POSITIVE_WORDS_PATH, 'r')
    n = open(NEGATIVE_WORDS_PATH, 'r')

    positives = [x[:-1] for x in p.readlines()]
    negatives = [x[:-1] for x in n.readlines()]


    positives = prep(positives)
    negatives = prep(negatives)

    similar_pos = set(positives)
    similar_neg = set(negatives)

    for w in positives:
        similar_pos.update([x[0] for x in word_vectors.most_similar(positive=[w])] if w in word_vectors else [])
    for w in negatives:
        similar_neg.update([x[0] for x in word_vectors.most_similar(positive=[w])] if w in word_vectors else [])

    return (similar_pos, similar_neg)


# Get similar positive and negative words
similar_pos, similar_neg = get_similar_words()

def create_dictionary(all_tokens, word_vectors):
    return {k:word_vectors[k] for k in all_tokens if k in word_vectors}


def create_average_vectors(tokens, vect_dict):
    # Returns average of the vectors of the given tokens
    average = np.array([])
    list_of_vectors = np.asarray([vect_dict[x] for x in tokens if x in vect_dict])

    if list_of_vectors.size != 0:
        average = np.mean(list_of_vectors, axis = 0)

    return average


def calc_pos_neg(tokens, word_set):
    i = 0
    for t in tokens:
        if t in word_set:
            i+=1
    return i/len(tokens)
 

def calc_positivity(tokens):
    return calc_pos_neg(tokens, similar_pos)    


def calc_negativity(tokens):
    return calc_pos_neg(tokens, similar_neg)


def partition_data(data):
    X = np.asarray([x[0] for x in data])
    y = np.asarray([x[1] for x in data])
    return (X, y)

def get_dataset(all_tokens, all_data):
    global word_vectors
    vect_dict = create_dictionary(all_tokens, word_vectors)
    data = []
    for d in all_data:
        avg_vec = create_average_vectors(d[0], vect_dict)
        if avg_vec.size != 0:
            avg_vec = np.append(avg_vec, calc_positivity(d[0]))
            avg_vec = np.append(avg_vec, calc_negativity(d[0]))
            data.append((avg_vec, d[1]))
    shuffle(data)
    return partition_data(data)

def main():
    with open(PREPROCESSED_TRAINING_FILE_PATH, 'rb') as f_read, open(DATASET_PATH, 'wb') as f_write:
        all_tokens, all_data = pickle.load(f_read)
        (X, y) = get_dataset(all_tokens, all_data)
        pickle.dump((X, y), f_write)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- {} seconds ---".format(end - start))
