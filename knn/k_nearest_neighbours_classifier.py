import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

genre_names = ['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


class kNearestNeighboursClassifier:
    def __init__(self):
        pass

    def train(self, training_set, training_labels):
        self.training_set = training_set
        self.training_labels = training_labels
        pass

    def classify(self, input, genres_to_classify=genre_names, distance=np.linalg.norm, k=1):
        # Inizialization:
        ## min_dist: this numpy array contains the minimum distances found in the program in increasing order
        ## class_dist: this numpy array contains the class associated to the distances saved in min_dist
        ## class_counter: this numpy array will be used at the end of the code to count how many times a genres is found between the minimum distances
        min_dist = np.array([2125262800.0 for i in range(k)])
        class_dist = np.array([-1.0 for i in range(k)])
        class_counter = np.array([0.0 for i in range(10)])

        # Iterating on the genres gave as input
        for genre_name in genres_to_classify:
            # Keeping the number value of the genre currently considered
            class_curr = genre_names.index(genre_name)

            # Here we extract the features of the training set that belong to the genre previously calculated
            # (the field class is then removed)
            analyzing_data = self.training_set[self.training_labels == class_curr, :]

            # Iterating on the data of the training set of the genre considered
            for i in range(analyzing_data.shape[0]):
                # Calculating the distance of the training set's data from the input.
                curr_dist = distance(input - analyzing_data[i, :])

                # If the distance found is minor than the maximum distance stored,
                # it should be inserted in the min_dist array
                if curr_dist < np.max(min_dist):
                    # Ranging inside the min_dist array to inserted the value
                    for j in range(k):
                        # If a value bigger than the distance currently considered is found,
                        # it means that the distance must go before that value.
                        if min_dist[j] > curr_dist:
                            # Inserting the value in the right place and cutting the min_dist array.
                            min_dist = np.insert(min_dist, j, curr_dist)
                            class_dist = np.insert(class_dist, j, class_curr)
                            min_dist = min_dist[0:k]
                            class_dist = class_dist[0:k]
                            print(min_dist)
                            break

        # If the k distances that we have to store are more than one we should count the genres found,
        # otherwise the result is immediately calculated
        if k > 1:
            result_counter = 0
            last_found = np.array([])

            # Iterating on the class_dist array
            for i in range(k):
                # For each saved genre of class_dist array the class counter is incremented
                class_counter[class_dist[i].astype(int)] = class_counter[class_dist[i].astype(int)] + 1

            # Iterating on class_counter
            for i in range(10):
                # This control is needed to check if there are more classes with the same number of nearest points
                if class_counter[i] == np.max(class_counter):
                    result_counter += 1
                    last_found = np.append(last_found, i)

            # If more genres have the same number of nearest points, the nearest genre is considered,
            # else the genre with maximum number of near
            # points is returned
            if result_counter > 1:
                # Iterating on class_dist
                for ii in range(k):
                    # Iterating on last_found
                    for jj in range(last_found.size):
                        # First nearest genre found
                        if last_found[jj] == class_dist[ii]:
                            return genre_names[class_dist[ii].astype(int)]
            else:
                return genre_names[last_found[0].astype(int)]

        else:
            return genre_names[int(class_dist)]

    def confusion_matrix(self, validation_set, validation_labels, genres_to_classify=genre_names, distance=np.linalg.norm, k = 1):
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
            predicted = self.classify(input_vector, distance= distance, genres_to_classify = genres_to_classify, k = k)

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
        return kNearestNeighboursClassifier.compute_accuracy(conf_mat)