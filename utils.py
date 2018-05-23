bad_chars = '.:",;()\'<>^&#'
FOLDER_PATH = './data/'
PREPROCESSED_TRAINING_FILE_PATH = FOLDER_PATH + "preprocessed_data"
WORD2VEC_MODEL_FILE_PATH = FOLDER_PATH + "word2vec_model"
DATASET_PATH = FOLDER_PATH + "dataset"
POSITIVE_WORDS_PATH = FOLDER_PATH + "positives"
NEGATIVE_WORDS_PATH = FOLDER_PATH + "negatives"
ML_MODEL_PATH = FOLDER_PATH + "predictor_model"
WIKI_FILE_PATH = FOLDER_PATH + "trwiki-20180101-pages-articles.xml.bz2"
TEST_SET_PATH = "test"

f = open('word_forms_stems_and_frequencies_full.txt', 'r')
WORDS = {x.split()[0]:int(x.split()[-1]) for x in f.readlines() if not x.startswith('#') and x != '\n'}

