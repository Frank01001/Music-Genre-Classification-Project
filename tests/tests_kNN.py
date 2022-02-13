import pandas as pd
from knn.k_nearest_neighbours_classifier import *
from feat_ext.sample_processing import get_normalized_train_valid_sets

genre_names = ['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Dataset import
path_to_extracted_dataset = ''
data = pd.read_csv(path_to_extracted_dataset)

training_set, training_labels, valid_set, valid_labels = get_normalized_train_valid_sets(data)

classifier = kNearestNeighboursClassifier()

classifier.train(training_set, training_labels)

confusion_matrix = classifier.confusion_matrix(valid_set, valid_labels)

print(confusion_matrix)

accuracy = kNearestNeighboursClassifier.compute_accuracy_from_matrix(confusion_matrix)

print('Accuracy %1.2f' % accuracy)

