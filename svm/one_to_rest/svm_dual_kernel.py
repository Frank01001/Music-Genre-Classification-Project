from svm.one_to_rest.svm_otr_classifier import *
import numpy as np
import pandas as pd

genre_names = ['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

class DualKernelClassifier(SVMClassifierOTR):
  def __init__(self, genre, dataset, labels):
    super().__init__(genre)
    indices = pd.read_csv('.\svm\one_to_rest\indexes_kernel.csv').to_numpy()[:, 1:]
    indices = indices.reshape((800,))

    self.dataset = dataset[indices, :]
    self.labels = labels[indices]

    B = pd.read_csv('.\svm\one_to_rest\B_kernel.csv').to_numpy()[:, 1]
    C = pd.read_csv('.\svm\one_to_rest\C_kernel.csv').to_numpy()[:, 1:]
    self.b = B[genre_names.index(self.genre)]
    self.c = C[genre_names.index(self.genre), :]


  def classify_isgenre(self, input):
    N= self.dataset.shape[0]
    genre_index = genre_names.index(self.genre)

    in_labels = np.zeros(N)
    in_labels[self.labels == genre_index] = 1
    in_labels[self.labels != genre_index] = -1

    w_phi = np.sum(np.array([self.c[i] * in_labels[i] * self.kernel_vec(input, self.dataset[i, :]) for i in range(N)]))

    return self.internal_classify(w_phi, self.b) > 0, self.internal_classify(w_phi, self.b)

  def train(self, x, y):
    raise Exception('This method is not implemented in this class as it would take to long. Please go to the notebook '
                    'NAML_Project_SVM_Multiclass_OneToRest to see how the train is done and compute it again.')

  def internal_classify(self, w, b):
    return w - b

  # This is the kernel function for the vectors
  def kernel_vec(self, xi, xj):
    return (np.dot(xi, xj) + 1) ** 3