from svm.one_to_one.multiclass_oto_classifier import *
import pandas as pd
from svm.one_to_rest.multiclass_otr_classifier import *
from feat_ext.sample_processing import get_normalized_train_valid_sets

genre_names = ['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Dataset import
data = pd.read_csv('C:\\Users\\frapa\\PycharmProjects\\namlProject14-21\\extracted_dataset.csv')

training_set, training_labels, valid_set, valid_labels = get_normalized_train_valid_sets(data)

# All 10 genres
test_classifier_10 = MulticlassSVM_OTO(genre_names, PrimalFeatureMapClassifier)
test_classifier_10.train_all(training_set, training_labels)

conf_mat = test_classifier_10.confusion_matrix(valid_set, valid_labels)
print(conf_mat)
print(MulticlassSVM_OTO.accuracy_from_matrix(conf_mat))

# 6 genres
test_classifier_6 = MulticlassSVM_OTO(['blues', 'classical', 'country', 'disco', 'pop', 'rock'], PrimalFeatureMapClassifier)
test_classifier_6.train_all(training_set, training_labels)

conf_mat = test_classifier_6.confusion_matrix(valid_set, valid_labels)
print(conf_mat)
print(MulticlassSVM_OTO.accuracy_from_matrix(conf_mat))

# 2 genres
test_classifier_2 = MulticlassSVM_OTO(['classical', 'pop'], PrimalFeatureMapClassifier)
test_classifier_2.train_all(training_set, training_labels)

conf_mat = test_classifier_2.confusion_matrix(valid_set, valid_labels)
print(conf_mat)
print(MulticlassSVM_OTO.accuracy_from_matrix(conf_mat))

