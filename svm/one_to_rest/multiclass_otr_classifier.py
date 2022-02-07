from svm.one_to_rest.svm_dual_kernel import *

class MulticlassSVM_OTO:
    def __init__(self, genre_list):
        self.active_genres = genre_list.copy()

        self.classifiers = list()

        for genre in genre_list:
            self.classifiers.append(DualKernelClassifier(genre))

    def divide_training_set(self, dataset, labels):
        indices = pd.read_csv('./indexes_kernel.csv')
        indices_valid = np.setdiff1d(np.arange(dataset.shape[0]), indices)
        np.random.shuffle(indices_valid)
        return dataset[indices, :], labels[indices, :], dataset[indices_valid, :], labels[indices_valid, :]

    #Input data = train dataset, input labels = train labels
    def load_all(self, input_data, input_labels):
        for i, svm_classif in enumerate(self.classifiers):
            print('Started Loading for classifier {}, with genre {}'.format(i + 1, svm_classif.genre))
            #The genre cleaning is done in the classifier
            svm_classif.load_from_files(input_data, input_labels, self.active_genres)
            print('Ended Loading for classifier {}, with genre {}'.format(i + 1, svm_classif.genre))

    def classify(self, input):
        classifications = -1000 * np.ones(len(genre_names))

        for classifier in self.classifiers:
            genre_index = genre_names.idex(classifier.genre)
            classifications[genre_index] = classifier.classify_isgenre(input)[1]

        return genre_names[np.argmax(classifications)]


    def confusion_matrix(self, validation_data, validation_labels):
        confusion_matrix = np.zeros((len(genre_names), len(genre_names)))

        mask = [False for i in range(validation_data.shape[0])]
        for genre in self.active_genres:
            mask = np.logical_or(mask, validation_labels == genre)
        validation_data = validation_data[mask]
        validation_labels = validation_labels[mask]

        for i in range(validation_data.shape[0]):
            predicted = genre_names.index(self.classify(validation_data[i, :]))
            confusion_matrix[validation_labels[i].astype(int), predicted] += 1

        return confusion_matrix


    def compute_accuracy(self, input_data, input_labels):
        conf_mat = self.confusion_matrix(input_data, input_labels)
        return (conf_mat.trace() / conf_mat.sum())