import sys
import time
import numpy as np
import pickle
from TurkishStemmer import TurkishStemmer
from gensim.models import Word2Vec
from random import shuffle

start_time = time.time()
path = './train/'

all_tokens = set()

model = Word2Vec.load('./model_data')
word_vectors = model.wv

p = open('positives', 'r')
n = open('negatives', 'r')

positives = [x[:-1] for x in p.readlines()]
negatives = [x[:-1] for x in n.readlines()]

def prep(word_list):
	stemmer = TurkishStemmer()
	word_list = [stemmer.stem(x) for x in word_list]
	word_list = [x.replace('ğ','g').replace('ı','i').replace('ç','c').replace('ş','s').replace('ü','u').replace('ö','o') for x in word_list]
	return word_list

positives = prep(positives)
negatives = prep(negatives)

similar_pos = set(positives)
similar_neg = set(negatives)

for w in positives:
	similar_pos.update([x[0] for x in word_vectors.most_similar(positive=[w])] if w in word_vectors else [])
for w in negatives:
	similar_neg.update([x[0] for x in word_vectors.most_similar(positive=[w])] if w in word_vectors else [])


def create_dictionary():
	global all_tokens
	return {k:word_vectors[k] for k in all_tokens if k in word_vectors}

def create_average_vectors(tokens, vect_dict):
	average = np.array([])
	# print (type(word_vectors))
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

all_data = []
with open('prep_data', 'rb') as f:
    all_tokens, all_data = pickle.load(f)
vect_dict = create_dictionary()
data = []
for d in all_data:
	avg_vec = create_average_vectors(d[0], vect_dict)
	if avg_vec.size != 0:
		avg_vec = np.append(avg_vec, calc_positivity(d[0]))
		avg_vec = np.append(avg_vec, calc_negativity(d[0]))
		data.append((avg_vec, d[1]))
shuffle(data)
train_data = data[:9*len(data)//10]
test_data = data[9*len(data)//10:]
X_train = np.asarray([x[0] for x in train_data])
Y_train = np.asarray([x[1] for x in train_data])
X_test = np.asarray([x[0] for x in test_data])
Y_test = np.asarray([x[1] for x in test_data])
with open('train_file', 'wb') as f:
    pickle.dump((X_train, Y_train, X_test, Y_test), f)

print("--- %s seconds ---" % (time.time() - start_time))
