import numpy as np
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm

data = None
with open('train_file', 'rb') as f:
    data = pickle.load(f)

X_train, Y_train, X_test, Y_test = data

print (X_train.shape)
print (Y_train.shape)
print (X_test.shape)
print (Y_test.shape)

classifier = OneVsRestClassifier(estimator=LogisticRegression())
classifier.fit(X_train, Y_train)

print (classifier.score(X_test, Y_test))

classifier = MLPClassifier(learning_rate = 'adaptive', activation = 'relu', hidden_layer_sizes = 10, early_stopping = True)
classifier.fit(X_train, Y_train)

print (classifier.score(X_test, Y_test))

classifier = svm.SVC()
classifier.fit(X_train, Y_train)

print (classifier.score(X_test, Y_test))