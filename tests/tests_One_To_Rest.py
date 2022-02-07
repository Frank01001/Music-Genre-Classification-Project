from svm.one_to_rest.multiclass_otr_classifier import *

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

classifier10 = MulticlassSVM_OTR(['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
classifier6 = MulticlassSVM_OTR(['blues', 'classical', 'country', 'disco', 'pop', 'rock'])
classifier2 = MulticlassSVM_OTR(['classical', 'pop'])

"""""

print('Small tests for the one to rest 10 genres classifier')
for i in range(3):
    print(classifier10.classify(dataset_valid[i, :]))

print('Small tests for the one to rest 6 genres classifier')
for i in range(3):
    print(classifier6.classify(dataset_valid[i, :]))
    
print('Small tests for the one to rest 4 genres classifier')
for i in range(3):
    print(classifier4.classify(dataset_valid[i, :]))

"""""


conf_mat = classifier10.internal_confusion_matrix()
print('Confusion matrix and accuracy of the 10 genres classifier')
print(conf_mat)
print(classifier10.internal_accuracy())

conf_mat = classifier6.internal_confusion_matrix()
print('Confusion matrix and accuracy of the 6 genres classifier')
print(conf_mat)
print(classifier6.internal_accuracy())

conf_mat = classifier2.internal_confusion_matrix()
print('Confusion matrix and accuracy of the 2 genres classifier')
print(conf_mat)
print(classifier2.internal_accuracy())

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