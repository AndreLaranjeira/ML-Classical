# Program to compare machine learning techniques on the MNIST dataset.

# Package imports:
import matplotlib.pyplot as plt
from mnist import MNIST
import pandas
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2

# User imports:
from image import center_of_gravity, image_matrix, preprocess_many

# Main function:

# First, we load the MNIST dataset with the 'python-mnist' package.
# This code is PROVIDED AT THE PACKAGE README FILE! We modified it a little, though.
mndata = MNIST(path='./data', return_type='numpy')
mndata.test_img_fname = 't10k-images.idx3-ubyte'
mndata.test_lbl_fname = 't10k-labels.idx1-ubyte'
mndata.train_img_fname = 'train-images.idx3-ubyte'
mndata.train_lbl_fname = 'train-labels.idx1-ubyte'

train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Extracting the input features:
train_features = preprocess_many(train_images)
test_features = preprocess_many(test_images)

# Running the LDA model analysis:
print("***********************")
print("LDA model:")
print("***********************", "\n")

LDA = LinearDiscriminantAnalysis()
LDA.fit(train_features, train_labels)
LDA_predictions = LDA.predict(test_features)

print("Accuracy: ", accuracy_score(test_labels, LDA_predictions))
print("Confusion matrix:\n", confusion_matrix(test_labels, LDA_predictions), '\n')
print("Classification report:\n", classification_report(test_labels, LDA_predictions))

# Running the KNN model analysis with K = 3:
print("***********************")
print("KNN with K = 3 model:")
print("***********************", "\n")

KNN3 = KNeighborsClassifier(n_neighbors = 3).fit(train_features, train_labels)
KNN3_predictions = KNN3.predict(test_features)

print("Accuracy:", accuracy_score(test_labels, KNN3_predictions))
print("Confusion matrix:\n", confusion_matrix(test_labels, KNN3_predictions), '\n')
print("Classification report:\n", classification_report(test_labels, KNN3_predictions))

# Running the KNN model analysis with K = 7:
print("***********************")
print("KNN with K = 7 model:")
print("***********************", "\n")

KNN7 = KNeighborsClassifier(n_neighbors = 7).fit(train_features, train_labels)
KNN7_predictions = KNN7.predict(test_features)

print("Accuracy:", accuracy_score(test_labels, KNN7_predictions))
print("Confusion matrix:\n", confusion_matrix(test_labels, KNN7_predictions), '\n')
print("Classification report:\n", classification_report(test_labels, KNN7_predictions))

# Running the KNN model analysis with K = 11:
print("***********************")
print("KNN with K = 11 model:")
print("***********************", "\n")

KNN11 = KNeighborsClassifier(n_neighbors = 11).fit(train_features, train_labels)
KNN11_predictions = KNN11.predict(test_features)

print("Accuracy:", accuracy_score(test_labels, KNN11_predictions))
print("Confusion matrix:\n", confusion_matrix(test_labels, KNN11_predictions), '\n')
print("Classification report:\n", classification_report(test_labels, KNN11_predictions))
