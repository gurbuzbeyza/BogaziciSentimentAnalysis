import sys
import time
import numpy as np
import keras
from nltk.tokenize import TweetTokenizer
from TurkishStemmer import TurkishStemmer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.corpora import WikiCorpus
from gensim import utils

start_time = time.time()
path = './train/'
bad_chars = '.:",;()\'<>^&#'

wiki_file = '../trwiki-20180101-pages-articles.xml.bz2'

all_tokens = []

def tokenize_and_stem(sentence,token_min_len=2,token_max_len=50,lower=True):
	global all_tokens
	stemmer = TurkishStemmer()
	tokenizer = TweetTokenizer()
	sentence = utils.to_unicode(sentence.lower())
	tokens = tokenizer.tokenize(sentence)
	tokens = [stemmer.stem(x.strip(bad_chars)) for x in tokens if x != '' and not x.startswith('@')]
	tokens = [x.replace('ğ','g').replace('ı','i').replace('ç','c').replace('ş','s').replace('ü','u').replace('ö','o') for x in tokens]
	all_tokens += tokens
	return tokens

def read_file(file_name, sentiment):
	f = open(path+file_name, 'r')
	lines = f.readlines()
	sentences = [x.split('\t\t\t')[1] for x in lines]
	tokenized_sentences = [[tokenize_and_stem(s),sentiment] for s in sentences]
	return tokenized_sentences

def word2vec_conversion(data):
	global all_tokens
	sentences = [x[0] for x in data]
	all_tokens = list(set(all_tokens))
	model = Word2Vec.load('./model_data')
	word_vectors = model.wv
	print (len(all_tokens))
	i = 0
	all_vectors = {}
	unrelevants = []
	for t in all_tokens:
		try:
			all_vectors[t] = word_vectors[t]
		except Exception as e:
			i+=1
			unrelevants.append(t)
	print (i)
	print(unrelevants)
	return [[[all_vectors[y] for y in x[0] if y in all_vectors], x[1]] for x in data]

all_data = read_file('positive-train',1) + read_file('notr-train',0) + read_file('negative-train',-1)
train_data = all_data[:9*len(all_data)//10]
test_data = all_data[9*len(all_data)//10:]
word2vec_conversion(train_data)
# print (word2vec_conversion(train_data)[0])

print("--- %s seconds ---" % (time.time() - start_time))
