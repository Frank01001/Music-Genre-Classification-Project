from svm.one_to_rest.svm_otr_classifier import *
import numpy as np
import scipy.optimize as opt
import svm.one_to_rest.OTR_utils as utils
import jax

genre_names = ['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

class DualKernelClassifier(SVMClassifierOTR):
  def __init__(self, genre):
    super().__init__(genre)
    self.b = None
    self.c = None

  def train(self, dataset_train, labels_train):
    N = dataset_train.shape[0]
    genre_index = genre_names.index(self.genre)

    # The results of the dual problem computed
    self.c = np.zeros(N)

    # Data of the genre considered
    class_genre = dataset_train[labels_train == genre_index, :]

    # Data of the other genres
    class_others = dataset_train[labels_train != genre_index, :]

    # Data and labels used in the resolution of the dual problem
    in_data = np.concatenate((class_genre, class_others), axis=0)
    in_labels = np.concatenate((np.ones(class_genre.shape[0]), -1 * np.ones(class_others.shape[0])), axis=None)

    utils.N_utils = N
    utils.in_labels = in_labels
    utils.in_data = in_data

    a = np.zeros(N)
    obj_k_jit = jax.jit(utils.obj_kernel)

    linear_constraint = opt.LinearConstraint(in_labels, 0, 0, keep_feasible=True)
    res = opt.minimize(obj_k_jit, a, method='trust-constr', jac='2-point', hess=utils.hessian,
                       constraints=[linear_constraint], options={'maxiter': 1000},
                       bounds=opt.Bounds(np.zeros(N), np.ones(N) * np.inf))

    self.c = np.array(res.x)

    index_non_zero = -1
    for j in range(N):
      if self.c[j] > 0 and index_non_zero < 0:
        index_non_zero = j
        break

    w_phi = np.sum(np.array([self.c[j] * in_labels[j] * self.kernel_vec(dataset_train[index_non_zero, :], dataset_train[j, :]) for j in range(N)]))

    self.b = - in_labels[index_non_zero] + w_phi

  """
  This method returns 2 objects that are:
    - A boolean that tells if the input is classified as the genre of the classifier
    - A float that represents the score calculated
  """
  def classify_isgenre(self, input, dataset, labels):
    N = self.dataset.shape[0]
    genre_index = genre_names.index(self.genre)

    in_labels = np.zeros(N)
    in_labels[labels == genre_index] = 1
    in_labels[labels != genre_index] = -1

    w_phi = np.sum(np.array([self.c[i] * in_labels[i] * self.kernel_vec(input, dataset[i, :]) for i in range(N)]))

    return w_phi - self.b > 0, w_phi - self.b

  # This is the kernel function for the vectors
  def kernel_vec(self, xi, xj):
    return (np.dot(xi, xj) + 1) ** 3