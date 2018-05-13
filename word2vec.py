import sys
import time
import multiprocessing
from nltk.tokenize import TweetTokenizer
from TurkishStemmer import TurkishStemmer
from gensim.corpora import WikiCorpus
from gensim import utils
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

bad_chars = '.:",;()\'<>^&#'
wiki_file = '../trwiki-20180101-pages-articles.xml.bz2'
tokenized_wiki_file = './tokenized_wiki_file'
def tokenize_and_stem(sentence,token_min_len=2,token_max_len=50,lower=True):
	stemmer = TurkishStemmer()
	tokenizer = TweetTokenizer()
	sentence = utils.to_unicode(sentence.lower())
	tokens = tokenizer.tokenize(sentence)
	tokens = [stemmer.stem(x.strip(bad_chars)) for x in tokens if x != '' and not x.startswith('@')]
	tokens = [x.replace('ğ','g').replace('ı','i').replace('ç','c').replace('ş','s').replace('ü','u').replace('ö','o') for x in tokens]
	return tokens

wiki = WikiCorpus(wiki_file, lemmatize=False,tokenizer_func = tokenize_and_stem)
output = open(tokenized_wiki_file,"w",encoding="utf-8")
i = 0
for text in wiki.get_texts():
	output.write(" ".join(text)+"\n")
	i+=1
model = Word2Vec(LineSentence(tokenized_wiki_file), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
model.save('model_data')