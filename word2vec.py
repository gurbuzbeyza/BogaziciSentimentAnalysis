'''
This script trains word2vec model on wikipedia dump and saves the trained model
'''

import sys
import time
import multiprocessing # for training the model on multiple cpu's
from nltk.tokenize import TweetTokenizer
from TurkishStemmer import TurkishStemmer
from gensim.corpora import WikiCorpus

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
# characters to be removed from tweets
from utils import bad_chars, WORD2VEC_MODEL_FILE_PATH, WIKI_FILE_PATH, TOKENIZED_WIKI_FILE_PATH
from gensim import utils

def tokenize_and_stem(sentence,token_min_len=2,token_max_len=50,lower=True):
	'''Tokenizes the given sentence and applies stemmer on each token'''
	stemmer = TurkishStemmer()
	tokenizer = TweetTokenizer()
	sentence = utils.to_unicode(sentence.lower())
	tokens = tokenizer.tokenize(sentence)
	tokens = [stemmer.stem(x.strip(bad_chars)) for x in tokens if x != '' and not x.startswith('@')]
	tokens = [x.replace('ğ','g').replace('ı','i').replace('ç','c').replace('ş','s').replace('ü','u').replace('ö','o') for x in tokens]
	return tokens

def main():
	# Load wikipedia data
	print("... Load wikipedia data")
	wiki = WikiCorpus(WIKI_FILE_PATH, lemmatize=False,tokenizer_func = tokenize_and_stem)

	# Save the wikipedia data before word2vec training, in case of any erros in the training phase
	print("... Save tokenized data")
	with open(TOKENIZED_WIKI_FILE_PATH,"w",encoding="utf-8") as output_file:
		for text in wiki.get_texts():
			output_file.write(" ".join(text)+"\n")

	# Train word2vec model and save it do disk
	print("... Train word2vec model")
	model = Word2Vec(LineSentence(TOKENIZED_WIKI_FILE_PATH), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
	model.save(WORD2VEC_MODEL_FILE_PATH)

if __name__ == "__main__":
	start = time.time()
	main()
	end = time.time()
	print("--- {} seconds ---".format(end - start))