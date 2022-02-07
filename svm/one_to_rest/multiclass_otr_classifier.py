from svm.one_to_rest.svm_dual_kernel import *

class MulticlassSVM_OTR:
    def __init__(self, genre_list):
        self.active_genres = genre_list.copy()

        self.classifiers = list()

        # Dataset import
        data = pd.read_csv('extracted_dataset.csv')

        # Dataset normalization
        data_mean = data.mean()
        data_std = data.std()

        data_normalized = (data - data_mean) / data_std

        self.dataset = data_normalized.to_numpy()[:, 1:4]
        self.labels = data.to_numpy()[:, 4]

        for genre in genre_list:
            self.classifiers.append(DualKernelClassifier(genre, self.dataset, self.labels))

    def classify(self, input):
        classifications = -1000 * np.ones(len(genre_names))

        for classifier in self.classifiers:
            genre_index = genre_names.index(classifier.genre)
            classifications[genre_index] = classifier.classify_isgenre(input)[1]

        return genre_names[np.argmax(classifications)]

    def internal_confusion_matrix(self):
        indices = pd.read_csv('.\svm\one_to_rest\indexes_kernel.csv').to_numpy()[:, 1:]
        indices_valid = np.setdiff1d(np.arange(1000), indices)
        np.random.shuffle(indices_valid)

        indices_valid = indices_valid.reshape((200,))
        validation_data = self.dataset[indices_valid, :]
        validation_labels = self.labels[indices_valid]
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
        return (conf_mat.trace() / conf_mat.sum())

    def internal_accuracy(self):
        conf_mat = self.internal_confusion_matrix()
        return (conf_mat.trace() / conf_mat.sum())