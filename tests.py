from svm.one_to_one.multiclass_oto_classifier import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

genre_names = ['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Dataset import
data = pd.read_csv('extracted_dataset.csv')

# Dataset normalization
data_mean = data.mean()
data_std = data.std()

data_normalized = (data - data_mean) / data_std

dataset = data_normalized.to_numpy()[:, 1:4]
labels = data.to_numpy()[:, 4].astype(int)

#Indexes extraction
indices = np.random.choice(1000, 1000, replace = False)
N_train = 800

indices_train = indices[:N_train]
indices_valid = indices[N_train:]

dataset_train = dataset[indices_train, :]
dataset_valid = dataset[indices_valid, :]

labels_train = labels[indices_train]
labels_valid = labels[indices_valid]

test_classifier = MulticlassSVM_OTO(genre_names, PrimalFeatureMapClassifier)
test_classifier.train_all(dataset_train, labels_train)

conf_mat = test_classifier.confusion_matrix(dataset_valid, labels_valid)
print(conf_mat)
print(MulticlassSVM_OTO.accuracy_from_matrix(conf_mat))
