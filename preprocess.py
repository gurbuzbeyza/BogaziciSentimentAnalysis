import numpy as np
import pickle
from nltk.tokenize import TweetTokenizer
from TurkishStemmer import TurkishStemmer
from gensim.models import Word2Vec
from gensim import utils

start_time = time.time()
path = './train/'
bad_chars = '.:",;()\'<>^&#'

wiki_file = '../trwiki-20180101-pages-articles.xml.bz2'

all_tokens = set()

f = open('word_forms_stems_and_frequencies_full.txt', 'r')
WORDS = {x.split()[0]:int(x.split()[-1]) for x in f.readlines() if not x.startswith('#') and x != '\n'}

def tokenize_and_stem(sentence,token_min_len=2,token_max_len=50,lower=True):
    global all_tokens
    stemmer = TurkishStemmer()
    tokenizer = TweetTokenizer()
    sentence = utils.to_unicode(sentence.lower())
    tokens = [stemmer.stem(correction(x.strip(bad_chars))) for x in tokenizer.tokenize(sentence) if x != '' and not x.startswith('@')]
    tokens = [x.replace('ğ','g').replace('ı','i').replace('ç','c').replace('ş','s').replace('ü','u').replace('ö','o') for x in tokens]
    all_tokens.update(tokens)
    return tokens


def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N if word in WORDS else 0

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def read_file(file_name, sentiment):
    f = open(path+file_name, 'r')
    lines = f.readlines()
    sentences = [x.split('\t\t\t')[1] for x in lines]
    dataset = [(tokenize_and_stem(s),sentiment) for s in sentences]
    return dataset

all_data = read_file('positive-train',1) + read_file('notr-train',0) + read_file('negative-train',-1)
with open('prep_data', 'wb') as f:
    pickle.dump((all_tokens, all_data), f)
print (all_tokens)