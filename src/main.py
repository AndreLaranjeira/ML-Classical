# Program to compare machine learning techniques on the MNIST dataset.

# Package imports:
import matplotlib.pyplot as plt
from mnist import MNIST
import pandas
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# User imports:
from image import image_matrix

# Main function:

# First, we load the MNIST dataset with the python-mnist package.
mndata = MNIST('./data')
mndata.test_img_fname = 't10k-images.idx3-ubyte'
mndata.test_lbl_fname = 't10k-labels.idx1-ubyte'
mndata.train_img_fname = 'train-images.idx3-ubyte'
mndata.train_lbl_fname = 'train-labels.idx1-ubyte'

train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Extracting the input features:

# Configuring parameters to run machine learning analyses:

# Running the LDA model analysis:

# Running the KNN model analysis with K = 3:

# Running the KNN model analysis with K = 7:

# Running the KNN model analysis with K = 11:
