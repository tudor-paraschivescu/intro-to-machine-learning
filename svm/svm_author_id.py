#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from sklearn import svm
from time import time
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Create and train the SVM classifier
clf = svm.SVC(kernel="linear")
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

# Make predictions for the test set
t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

# Calculate accuracy
accuracy = accuracy_score(labels_test, pred)
print "SVM author identifier accuracy: %f" % accuracy
