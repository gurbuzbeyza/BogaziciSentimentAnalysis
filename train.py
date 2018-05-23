'''This script applies some clustering algorithms to data and returns the score for each algorithm'''
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn import metrics
import numpy as np
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
import time
from sklearn.model_selection import train_test_split
from utils import DATASET_PATH, ML_MODEL_PATH

data = None
with open(DATASET_PATH, 'rb') as f:
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

    scores = cross_validate(clf, X, y, cv = 10, n_jobs = -1,  scoring=scoring)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
    print("Macro Precision: %0.2f (+/- %0.2f)" % (scores['test_macro_precision'].mean(), scores['test_macro_precision'].std() * 2))
    print("Macro Recall: %0.2f (+/- %0.2f)" % (scores['test_macro_recall'].mean(), scores['test_macro_recall'].std() * 2))
    print("Macro F1: %0.2f (+/- %0.2f)" % (scores['test_macro_f1_score'].mean(), scores['test_macro_f1_score'].std() * 2))
    print("Micro Precision: %0.2f (+/- %0.2f)" % (scores['test_micro_precision'].mean(), scores['test_micro_precision'].std() * 2))
    print("Micro Recall: %0.2f (+/- %0.2f)" % (scores['test_micro_recall'].mean(), scores['test_micro_recall'].std() * 2))
    print("Micro F1: %0.2f (+/- %0.2f)" % (scores['test_micro_f1_score'].mean(), scores['test_micro_f1_score'].std() * 2))
    print('---------------------')

def compareClassifiers():
    classifiers = []

    classifier_names = ['One vs Rest Logistic Regression', 'Multilayer Perceptron', 'SVM', 'Linear SVM']

    classifiers.append(OneVsRestClassifier(estimator=LogisticRegression()))

    classifiers.append(MLPClassifier(learning_rate = 'adaptive', activation = 'relu', hidden_layer_sizes = 10, early_stopping = True))

    classifiers.append(svm.SVC())

    classifiers.append(svm.LinearSVC())

    for clf, clf_name in zip(classifiers, classifier_names):
        calculateScores(clf, clf_name)

def main():
    # Logistic Regression is the best model according to the metrics. Therefore we train a Logistic Regression
    # model and save the parameters.
    clf = LogisticRegressionCV(cv = 10 , penalty = 'l2', solver = 'lbfgs', n_jobs=-1, refit = True, multi_class = "multinomial")

    # Split the dataset into training and set
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25 , shuffle = True )

    # Fit the model
    clf.fit(X_train, y_train)

    # Save model
    with open(ML_MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)

    print("Score of the model : {}".format(clf.score(X_test, y_test)))

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- {} seconds ---".format(end - start))