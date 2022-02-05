from abc import abstractmethod


class SVMClassifierOTR:
  def __init__(self, genre):
    self.genre = genre # Corresponds to SVM label 1

  @abstractmethod
  def bin_classify(self, input):
    pass

  @abstractmethod
  def train(self, x, y):
    pass

