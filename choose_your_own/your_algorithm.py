#!/usr/bin/python

import matplotlib.pyplot as plt
import sklearn.ensemble as ensemble
import sklearn.tree as tree
from sklearn.metrics import accuracy_score
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

ATTEMPTS = 100

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################
accuracy_sum = 0
max_accuracy = 0

# create classifier
clf = ensemble.RandomForestClassifier(n_estimators=2, max_features=None, min_samples_split=7)

for _ in xrange(1, ATTEMPTS):
    # retrain the classifier
    clf.fit(features_train, labels_train)

    # predict the test labels
    predictions = clf.predict(features_test)

    # calculate the accuracy
    accuracy = accuracy_score(labels_test, predictions)

    # add the accuracy to the average
    accuracy_sum += accuracy

    # select the max accuracy
    max_accuracy = max(max_accuracy, accuracy)

print "Average accuracy of my classifier: %.3f%%" % (accuracy_sum * 100 / ATTEMPTS)
print "Maximum accuracy of my classifier: %.3f%%" % (max_accuracy * 100)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
