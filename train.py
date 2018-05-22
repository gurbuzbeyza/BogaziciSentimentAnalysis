'''This script applies some clustering algorithms to data and returns the score for each algorithm
'''
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn import metrics
import numpy as np
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm

data = None
with open('train_file', 'rb') as f:
    data = pickle.load(f)

X, y = data

def calculateScores(clf, clf_name):
	print ('Scores of classifier ' + clf_name)
	
	scoring = {'accuracy' : make_scorer(accuracy_score), 
           'macro_precision' : make_scorer(precision_score,average='macro'),
           'macro_recall' : make_scorer(recall_score,average='macro'), 
           'macro_f1_score' : make_scorer(f1_score,average='macro'), 
           'micro_precision' : make_scorer(precision_score,average='micro'),
           'micro_recall' : make_scorer(recall_score,average='micro'), 
           'micro_f1_score' : make_scorer(f1_score,average='micro')}

	scores = cross_validate(clf, X, y, cv = 10, scoring=scoring)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
	print("Macro Precision: %0.2f (+/- %0.2f)" % (scores['test_macro_precision'].mean(), scores['test_macro_precision'].std() * 2))
	print("Macro Recall: %0.2f (+/- %0.2f)" % (scores['test_macro_recall'].mean(), scores['test_macro_recall'].std() * 2))
	print("Macro F1: %0.2f (+/- %0.2f)" % (scores['test_macro_f1_score'].mean(), scores['test_macro_f1_score'].std() * 2))
	print("Micro Precision: %0.2f (+/- %0.2f)" % (scores['test_micro_precision'].mean(), scores['test_micro_precision'].std() * 2))
	print("Micro Recall: %0.2f (+/- %0.2f)" % (scores['test_micro_recall'].mean(), scores['test_micro_recall'].std() * 2))
	print("Micro F1: %0.2f (+/- %0.2f)" % (scores['test_micro_f1_score'].mean(), scores['test_micro_f1_score'].std() * 2))
	print('---------------------')


classifiers = []

classifier_names = ['One vs Rest Logistic Regression', 'Multilayer Perceptron', 'SVM', 'Linear SVM']

classifiers.append(OneVsRestClassifier(estimator=LogisticRegression()))

classifiers.append(MLPClassifier(learning_rate = 'adaptive', activation = 'relu', hidden_layer_sizes = 10, early_stopping = True))

classifiers.append(svm.SVC())

classifiers.append(svm.LinearSVC())

for clf, clf_name in zip(classifiers, classifier_names):
	calculateScores(clf, clf_name)