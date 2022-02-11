from abc import abstractmethod


class SVMClassifierOTR:
  def __init__(self, genre):
    self.genre = genre # Corresponds to SVM label 1

  @abstractmethod
  def classify_isgenre(self, input, dataset, labels):
    pass

  @abstractmethod
  def train(self, dataset_train, labels_train):
    pass

