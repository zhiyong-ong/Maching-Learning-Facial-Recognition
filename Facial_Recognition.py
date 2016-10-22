'''
Created on 21 Oct 2016

@author: zhiyong
'''
#  Images are single-channel with pixel intensity represented by a float value from 0 to 255, 
#  the single-channel intensity is the mean value of the original RGB channel intensities.
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


fo = open('testResults.txt', 'w')
# X_train.npy: training data numpy array. 
# Each row in the array corresponds to an image unrolled to a vector (50 x 37 = 1850 dimension)
X_train = np.load('C:/Users/zhiyong/Desktop/School Work/CS3244/HW3/X_train.npy')
  
# y_train.npy: labels (0-6) of each data corresponding to the image in the same row in X_train.npy
y_train = np.load('C:/Users/zhiyong/Desktop/School Work/CS3244/HW3/y_train.npy')

X_test = np.load('C:/Users/zhiyong/Desktop/School Work/CS3244/HW3/X_test.npy')

all_data = []
for i in range(len(y_train)):
    is_same = False
    for j in range(len(all_data)):
        if y_train[i] == all_data[j]:
            is_same = True
            break
    if is_same != True:
        all_data.append(y_train[i])




# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

n_classes = len(all_data)
n_samples = len(X_train)
n_features = len(X_train[0])

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

n_components = 150
h = 1
w = 1850
print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))
              
eigenfaces = pca.components_.reshape((n_components, h, w))
print("Projecting the input data on the eigenfaces orthonormal basis")
  
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("done in %0.3fs" % (time() - t0))

print(X_train_pca.shape)

print("Fitting the classifier to the training set")
t0 = time()

# Runs 1 vs 1 NOT 1 vs REST.
param_grid = {'C': [1, 3, 5, 1e1, 5e1, 1e2, 1e3],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(estimator=SVC(kernel='rbf', class_weight='balanced'), param_grid=param_grid, cv=10)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Error in")
internal_pred = clf.predict(X_train_pca)
print(classification_report(y_train, internal_pred))

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(y_pred)
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
# 
# def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
#     """Helper function to plot a gallery of portraits"""
#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
#     plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
#     for i in range(n_row * n_col):
#         plt.subplot(n_row, n_col, i + 1)
#         plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
#         plt.title(titles[i], size=12)
#         plt.xticks(())
#         plt.yticks(())
# 
# 
# # plot the result of the prediction on a portion of the test set
# 
# # plot the gallery of the most significative eigenfaces
# 
# eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
# plot_gallery(eigenfaces, eigenface_titles, h, w)
# 
# plt.show()
