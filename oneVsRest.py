from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.tree import DecisionTreeClassifier

#  Images are single-channel with pixel intensity represented by a float value from 0 to 255,
#  the single-channel intensity is the mean value of the original RGB channel intensities.
print("loading")
# X_train.npy: training data numpy array.
# Each row in the array corresponds to an image unrolled to a vector (50 x 37 = 1850 dimension)
X_train = np.load('X_train.npy')
# y_train.npy: labels (0-6) of each data corresponding to the image in the same row in X_train.npy
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')

# Generate testResults.csv
# Run this with y_pred

def saveTestResults(y_pred, filename='testResults.csv'):
    fo = open(filename, 'w')
    fo.write("ImageId,PredictedClass\n")
    for i in range(len(y_pred)):
        if i == len(y_pred) - 1:
            fo.write(str(i) + "," + str(y_pred[i]))
        else:
            fo.write(str(i) + "," + str(y_pred[i]) + "\n")
    fo.close()
    

# to calculate number of classes
def calculateClass(label):
    n_classes = []
    for n_class in label:
        if n_class not in n_classes:
            n_classes.append(n_class)
        else:
            continue
    return len(n_classes)

print("Total train data size:")
print("n_samples: %d" % len(X_train))
print("n_features: %d" % len(X_train[0]))
print("n_classes: %d" % calculateClass(y_train))

print("Total test data size:")
print("n_samples: %d" % len(X_test))
print("n_features: %d" % len(X_test[0]))

n_components = 150
h = 50
w = 37

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

eigenfaces = pca.components_.reshape((n_components, h, w))
print("Projecting the input data on the eigenfaces orthonormal basis")

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# GridSearchCV with SVC
print("Fitting the classifier to the training set")
      
model_to_set = OneVsRestClassifier(SVC(kernel="rbf"))

parameters = {
    "estimator__C": [1, 3, 5, 1e1, 5e1, 1e2, 1e3],
    "estimator__kernel": ["rbf"],
    "estimator__gamma":[0.0001, 0.0005,0.001,0.005,0.01,0.1]
}

clf = GridSearchCV(model_to_set, param_grid=parameters)
 
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Ein")
internal_pred = clf.predict(X_train_pca)
print(classification_report(y_train, internal_pred))

print("Eout")
y_pred = clf.predict(X_test_pca)
saveTestResults(y_pred)
