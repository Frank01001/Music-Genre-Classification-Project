import numpy as np


# Assumption (x already contains the difference between the label and the prediction)
def manhattan(x):
    return np.abs(x).sum()


# Assumption (x already contains the difference between the label and the prediction)
def chebyshev(x):
    return np.max(np.abs(x))
