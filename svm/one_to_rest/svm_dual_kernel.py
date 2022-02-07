from svm.one_to_rest.svm_otr_classifier import *
import numpy as np
import pandas as pd

genre_names = ['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

class DualKernelClassifier(SVMClassifierOTR):
  def __init__(self, genre):
    super().__init__(genre)
    self.labels = None
    self.dataset = None

  def classify_isgenre(self, input):
    N=self.dataset.shape[0]
    genre_index = genre_names.index(self.genre)

    in_labels = np.zeros(N)
    in_labels[self.labels == genre_index] = 1
    in_labels[self.labels != genre_index] = -1

    w_phi = np.sum(np.array([self.c[genre_index, i] * in_labels[i] * self.kernel_vec(input, self.dataset[i, :]) for i in range(N)]))
    b = self.b[genre_index]

    return self.internal_classify(w_phi,b ) > 0, self.internal_classify(w_phi,b )

  def train(self, x, y):
    raise Exception('This method is not implemented in this class as it would take to long. Please go to the notebook '
                    'NAML_Project_SVM_Multiclass_OneToRest to see how the train is done and compute it again.')

  def load_from_files(self, x, y, genres):
    B = pd.read_csv('./B_kernel.csv')
    C = pd.read_csv('./C_kernel.csv')
    self.b=B[genre_names.index(self.genre)]
    self.c = C[genre_names.index(self.genre), :]
    mask = [False for i in range(x.shape[0])]
    for genre in genres:
      mask = np.logical_or(mask, y == genre)
    self.dataset = x[mask]
    self.labels = y[mask]

  def internal_classify(self, w, b):
    return w - b

  # This is the kernel function for the vectors
  def kernel_vec(self, xi, xj):
    return (np.dot(xi, xj) + 1) ** 3