'''
This script applies preprocessing on training data and saves the final output
'''

import numpy as np
import pickle
from nltk.tokenize import TweetTokenizer
from TurkishStemmer import TurkishStemmer
from gensim.models import Word2Vec
# characters to be removed from tweets
from utils import correction, bad_chars
from gensim import utils

# Folder path for training files
path = './Train/'

all_tokens = set()
stemmer = TurkishStemmer()
tokenizer = TweetTokenizer()

def tokenize_and_stem(sentence):
    '''Tokenizes the given sentence and applies stemmer on each token'''
    global all_tokens, stemmer, tokenizer
    sentence = utils.to_unicode(sentence.lower())
    tokens = tokenizer.tokenize(sentence)
    tokens = [stemmer.stem(x.strip(bad_chars)) for x in tokens if x != '' and not x.startswith('@')]
    tokens = [x.replace('ğ','g').replace('ı','i').replace('ç','c').replace('ş','s').replace('ü','u').replace('ö','o') for x in tokens]
    all_tokens.update(tokens)
    return tokens

def read_file(file_name, sentiment):
    '''Reads file with the specified file_name and returns preprocessed data'''
    f = open(path+file_name, 'r')
    lines = f.readlines()
    sentences = [x.split('\t\t\t')[1] for x in lines]
    dataset = [(tokenize_and_stem(s),sentiment) for s in sentences]
    return dataset

def main():
    all_data = read_file('positive-train',1) + read_file('notr-train',0) + read_file('negative-train',-1)
    with open('preprocessed_data', 'wb') as f:
        pickle.dump((all_tokens, all_data), f)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- {} seconds ---".format(end - start))
