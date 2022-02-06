import numpy as np

genre_names = ['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


class NearestCentroidClassifier:
    def __init__(self):
        self.centroids = list()

    '''
    Computes the class centroids
    WARNING: dataset and labels are the COMPLETE dataset, even if you
    want to classify a less than 10 genres
    '''
    def train(self, dataset, labels):
        self.centroids = list()
        for class_index in range(len(genre_names)):
            # partial_set only contains the sample of a certain class
            partial_set = dataset[labels == class_index, :]
            # class_centroid contains the mean for each feature
            class_centroid = partial_set.mean(axis=0)
            self.centroids.append(class_centroid)

    def classify(self, input, genres_to_classify=genre_names, distance=np.linalg.norm, debug_mode=False):
        min_distance = np.infty
        current_closest = ''
        for i, genre in enumerate(genre_names):
            if genre not in genres_to_classify:
                continue

            current_centroid = self.centroids[i]
            buffer = distance(input - current_centroid)
            if buffer < min_distance:
                min_distance = buffer
                current_closest = genre

            if debug_mode:
                print('Sample distance from ' + genre + ' is ' + str(buffer))

        return current_closest

    def confusion_matrix(self, validation_set, validation_labels, genres_to_classify=genre_names, distance=np.linalg.norm):
        confusion_matrix = np.zeros((10, 10))

        # Indices of genres to classify
        genre_indices = list()
        for genre_n in genres_to_classify: genre_indices.append(genre_names.index(genre_n))

        # Filter to only select samples from the input genres
        test_set_filter = np.zeros(validation_set.shape[0], dtype='bool')
        for j in range(validation_set.shape[0]):
            test_set_filter[j] = validation_labels[j] in genre_indices

        # Application of filter on the test set
        partial_test_set = validation_set[test_set_filter, :]
        partial_test_labels = validation_labels[test_set_filter]

        # Size of the test set
        test_count = validation_set.shape[0]

        # For each test sample
        for test_index in range(test_count):
            # True label (index form)
            true_class_index = partial_test_labels[test_index]
            # True label (string form)
            true_class = genre_names[true_class_index]

            # Predict
            input_vector = partial_test_set[test_index, :]
            predicted = self.classify(input_vector, distance= distance, genres_to_classify = genres_to_classify)

            # Prediction (index form)
            predicted_index = genre_names.index(predicted)

            # Add result to confusion matrix
            confusion_matrix[true_class_index, predicted_index] += 1

        return confusion_matrix

    @staticmethod
    def compute_accuracy(c_mat):
        return c_mat.trace() / np.sum(c_mat) * 100.0

    def compute_accuracy(self):
        conf_mat = self.confusion_matrix()
        return NearestCentroidClassifier.compute_accuracy(conf_mat)