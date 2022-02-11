import pandas as pd
from feat_ext.sample_processing import get_normalized_train_valid_sets
from svm.one_to_rest.multiclass_otr_classifier import *

# Dataset import
data = pd.read_csv('C:\\Users\\Marino\\PycharmProjects\\NAMLProject14-21\\extracted_dataset.csv')

dataset_train, labels_train, dataset_valid, labels_valid = get_normalized_train_valid_sets(data)

classifier10 = MulticlassSVM_OTR(['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
classifier6 = MulticlassSVM_OTR(['blues', 'classical', 'country', 'disco', 'pop', 'rock'])
classifier2 = MulticlassSVM_OTR(['classical', 'pop'])

classifier10.train_all(dataset_train, labels_train)
classifier6.train_all(dataset_train, labels_train)
classifier2.train_all(dataset_train, labels_train)

conf_mat = classifier10.confusion_matrix(dataset_valid, labels_valid)
print('Confusion matrix and accuracy of the 10 genres classifier')
print(conf_mat)
print(classifier10.compute_accuracy(dataset_valid, labels_valid))

conf_mat = classifier6.confusion_matrix(dataset_valid, labels_valid)
print('Confusion matrix and accuracy of the 6 genres classifier')
print(conf_mat)
print(classifier6.compute_accuracy(dataset_valid, labels_valid))

conf_mat = classifier2.confusion_matrix(dataset_valid, labels_valid)
print('Confusion matrix and accuracy of the 2 genres classifier')
print(conf_mat)
print(classifier2.compute_accuracy(dataset_valid, labels_valid))