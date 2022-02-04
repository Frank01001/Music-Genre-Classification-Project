from svm.one_to_one.decision_tree import *
from svm.one_to_one.svm_primal_linear import *
from svm.one_to_one.svm_primal_featuremap import *


class MulticlassSVM:
    def __init__(self, genre_list, classifier_type : type):
        self.active_genres = genre_list.copy()

        n_gen = len(genre_list)

        self.classifiers = list()

        for i in range(n_gen):
            for j in range(i + 1, n_gen):
                genre1 = genre_list[i]
                genre2 = genre_list[j]

                if classifier_type == PrimalLinearClassifier:
                    classifier = PrimalLinearClassifier(genre1, genre2)
                elif classifier_type == PrimalFeatureMapClassifier:
                    classifier = PrimalFeatureMapClassifier(genre1, genre2)
                else:
                    raise TypeError("Unexpected type " + str(classifier_type))
                self.classifiers.append(classifier)

        self.decision_tree = DecisionBinaryTree(genre_list, self.classifiers)

    def train_all(self, input_data, input_labels):
        for i, svm_classif in enumerate(self.classifiers):
            print('Started Training for classifier {}, with genres {} and  {}'.format(i + 1, svm_classif.genre1,
                                                                                      svm_classif.genre2))
            svm_classif.train(input_data, input_labels)
            print('Ended Training for classifier {}, with genres {} and  {}'.format(i + 1, svm_classif.genre1,
                                                                                    svm_classif.genre2))

    def classify(self, input):
        # classification with binary decision tree
        return self.decision_tree.classify(input)

    def confusion_matrix(self, input_data, input_labels):
        active_genres_indices = [genre_names.index(genre) for genre in self.active_genres]
        allowed_samples = np.array([(input_labels[i] in active_genres_indices) for i in range(input_labels.size)])

        sub_data = input_data[allowed_samples, :]
        sub_labels = input_labels[allowed_samples]

        N_genres = len(self.active_genres)

        confusion_mat = np.zeros((10, 10))

        for test_index in range(sub_labels.size):
            sample = sub_data[test_index, :]
            label = sub_labels[test_index]

            predicted_str = self.classify(sample)
            predicted_i = genre_names.index(predicted_str)

            confusion_mat[label, predicted_i] += 1

        return confusion_mat

    def compute_accuracy(self, input_data, input_labels):
        conf_mat = self.confusion_matrix(input_data, input_labels)
        return MulticlassSVM.accuracy_from_matrix(conf_mat)

    # Utils
    @staticmethod
    def accuracy_from_matrix(conf_mat):
        return conf_mat.trace() / np.sum(conf_mat)
