from svm.one_to_rest.svm_dual_kernel import *

class MulticlassSVM_OTR:
    def __init__(self, genre_list):
        self.partial_training_labels = None
        self.partial_training_set = None
        self.active_genres = genre_list.copy()

        self.classifiers = list()

        for genre in genre_list:
            self.classifiers.append(DualKernelClassifier(genre))


    def train_all(self, dataset_train, labels_train):
        genres = self.active_genres

        # Indices of genres to classify
        genre_indices = list()
        for genre_n in genres: genre_indices.append(genre_names.index(genre_n))

        # Filter to only select samples from the input genres
        training_set_filter = np.zeros(dataset_train.shape[0], dtype='bool')
        for j in range(dataset_train.shape[0]):
            training_set_filter[j] = labels_train[j] in genre_indices

        # Application of filter on the test set
        self.partial_training_set = dataset_train[training_set_filter, :]
        self.partial_training_labels = labels_train[training_set_filter]

        for classifier in self.classifiers:
            classifier.train(self.partial_training_set, self.partial_training_labels)


    def classify(self, input):
        classifications = -1000 * np.ones(len(self.active_genres))

        for classifier in self.classifiers:
            genre_index = self.active_genres.index(classifier.genre)
            classifications[genre_index] = classifier.classify_isgenre(input, self.partial_training_set, self.partial_training_labels)[1]

        return self.active_genres[np.argmax(classifications)]


    def confusion_matrix(self, validation_data, validation_labels):
        confusion_matrix = np.zeros((len(genre_names), len(genre_names)))

        mask = [False for i in range(validation_data.shape[0])]
        for genre in self.active_genres:
            mask = np.logical_or(mask, validation_labels == genre_names.index(genre))

        validation_data = validation_data[mask, :]
        validation_labels = validation_labels[mask]

        for i in range(validation_data.shape[0]):
            gen_pred = self.classify(validation_data[i, :])
            predicted = genre_names.index(gen_pred)
            confusion_matrix[validation_labels[i].astype(int), predicted] += 1

        return confusion_matrix


    def compute_accuracy(self, input_data, input_labels):
        conf_mat = self.confusion_matrix(input_data, input_labels)
        return (conf_mat.trace() / conf_mat.sum())*100
