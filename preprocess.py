'''
This script applies preprocessing on training data and saves the final output
'''

import numpy as np
import pickle
from nltk.tokenize import TweetTokenizer
from TurkishStemmer import TurkishStemmer
from gensim.models import Word2Vec
# characters to be removed from tweets
from utils import  bad_chars, PREPROCESSED_TRAINING_FILE_PATH, TRAIN_POSITIVE_PATH, TRAIN_NEGATIVE_PATH, TRAIN_NOTR_PATH
from gensim import utils

# Folder path for training files
path = './Train/'

all_tokens = set()
stemmer = TurkishStemmer()
tokenizer = TweetTokenizer()

def tokenize_and_stem(sentence, all_tokens):
    '''Tokenizes the given sentence and applies stemmer on each token'''
    global stemmer, tokenizer
    sentence = utils.to_unicode(sentence.lower())
    tokens = tokenizer.tokenize(sentence)
    tokens = [stemmer.stem(x.strip(bad_chars)) for x in tokens if x != '' and not x.startswith('@')]
    tokens = [x.replace('ğ','g').replace('ı','i').replace('ç','c').replace('ş','s').replace('ü','u').replace('ö','o') for x in tokens]
    all_tokens.update(tokens)
    return tokens

def read_file(file_path, all_tokens,sentiment=None):
    '''Reads file with the specified file_name and returns preprocessed data'''
    f = open(file_path, 'r')
    lines = f.readlines()
    sentences = [x.split('\t\t\t')[1] for x in lines]
    if sentiment:
        dataset = [(tokenize_and_stem(s, all_tokens),sentiment) for s in sentences]
    else: # For test sets without sentiment
        dataset = [tokenize_and_stem(s, all_tokens) for s in sentences]
    return dataset

def get_preprocessed_data(file_names, sentiments = None):
    all_tokens = set()
    all_data = []
    if sentiments:
        for file_name, sentiment in zip(file_names, sentiments):
            all_data.append(read_file(file_name, all_tokens, sentiment))
    else:
        for file_name in file_names:
            all_data.append(read_file(file_name , all_tokens))
    return all_tokens, all_data

def main():
    all_tokens, all_data = get_preprocessed_data([TRAIN_POSITIVE_PATH, TRAIN_NOTR_PATH, TRAIN_NEGATIVE_PATH],[1,0,-1])
    with open(PREPROCESSED_TRAINING_FILE_PATH, 'wb') as f:
        pickle.dump((all_tokens, all_data), f)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- {} seconds ---".format(end - start))
