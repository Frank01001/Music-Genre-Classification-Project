from abc import abstractmethod


class SVMClassifier_OTO:
  def __init__(self, genre1, genre2):
    self.genre1 = genre1 # Corresponds to SVM label 1
    self.genre2 = genre2 # Corresponds to SVM label -1

  @abstractmethod
  def bin_classify(self, input):
    pass

  @abstractmethod
  def train(self, x, y):
    pass

