import pandas as pd
from knn.k_nearest_neighbours_classifier import *

genre_names = ['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Dataset import
data = pd.read_csv('../extracted_dataset.csv')

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

classifier = kNearestNeighboursClassifier()

classifier.train(dataset_train, labels_train)

confusion_matrix = classifier.confusion_matrix(dataset_valid, labels_valid)

print(confusion_matrix)

accuracy = kNearestNeighboursClassifier.compute_accuracy_from_matrix(confusion_matrix)

print('Accuracy %1.2f' % accuracy)

