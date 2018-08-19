#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
import numpy as np
from time import time
from sklearn.naive_bayes import GaussianNB
sys.path.append("../tools/")
from email_preprocess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Create and train the Naive Bayes classifier
clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

# Make predictions for the test set
t0 = time()
computed_labels = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

# Calculate accuracy
correct_answers = np.sum(computed_labels == labels_test)
accuracy = float(correct_answers) / len(labels_test)
print "Naive Bayes author identifier accuracy: %f" % accuracy
