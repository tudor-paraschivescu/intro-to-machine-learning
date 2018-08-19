#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy
import collections
from sklearn import svm
from time import time
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# small datasets that could be used to make computation 100 times faster (~1s)
small_features_train = features_train[:len(features_train)/100]
small_labels_train = labels_train[:len(labels_train)/100]

# Create the SVM classifier with different parameters
# In the rest of the code, the version with the best accuracy will be used
clf_linear = svm.SVC(kernel="linear")
clf_rbf = svm.SVC(kernel="rbf")
clf_rbf_c = svm.SVC(C=10000., kernel="rbf")

# Train the classifier
t0 = time()
clf_rbf_c.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

# Make predictions for the test set
t0 = time()
pred = clf_rbf_c.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

# Calculate accuracy
accuracy = accuracy_score(labels_test, pred)
print "SVM author identifier accuracy: %f" % accuracy

# Show the value predicted for the 10th, 26th and 50th feature
print "10: %d | 26: %d | 50: %d" % (pred[10], pred[26], pred[50])

# Show the number of predictions for Sara (0) and Chris (1)
print collections.Counter(pred)
