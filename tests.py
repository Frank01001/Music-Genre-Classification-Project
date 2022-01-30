import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

genre_names = ['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Dataset import
data = pd.read_csv('extracted_dataset.csv')

# Dataset normalization
data_mean = data.mean()
data_std = data.std()

data_normalized = (data - data_mean) / data_std

dataset = data_normalized.to_numpy()[:, 1:4]


class1_data = dataset[900:1000, :]
class2_data = dataset[100:200, :]

reg = 0.001
learning_rate = 0.02

w = np.random.randn(3)
b = 0

def classify(x):
  return (np.dot(w, x) - b)

def hingeloss(x, y):
  prod = y * classify(x)

  if prod >= 1: return 0
  else: return 1-prod


def hinge_derivative_w(x, y):
  prod = y * classify(x)
  prod_elem = y * x
  if prod >= 1: return np.zeros(x.shape[0])
  else: return -prod_elem

def hinge_derivative_b(x, y):
  prod = y * classify(x)
  if prod >= 1: return 0
  else: return y


history_loss = list()
loss = 0

for epoch in range(1000):
  # Computation of the gradient on the complete dataset
  gradient_w = np.zeros(w.size)
  gradient_b = 0

  for sample_index in range(100):
    gradient_w += hinge_derivative_w(class1_data[sample_index,:], 1)
    gradient_w += hinge_derivative_w(class2_data[sample_index,:], -1)
    gradient_b += hinge_derivative_b(class1_data[sample_index,:], 1)
    gradient_b += hinge_derivative_b(class2_data[sample_index,:], -1)

  gradient_w /= 200.0
  gradient_b /= 200.0

  # Gradient Descent step

  w = w - learning_rate * gradient_w
  b = b - learning_rate * gradient_b

# create x,y
xx, yy = np.meshgrid(range(-2,3), range(-2,3))

# calculate corresponding z
z = (-w[0] * xx - w[1] * yy + b) * 1. / w[2]

# plot the surface
plt.figure(figsize=(20,10))
plt3d = plt.figure(figsize=(20,10)).gca(projection='3d')
plt3d.plot_surface(xx, yy, z, alpha=0.2)


#and i would like to plot this point :
for i in range(100):
  plt3d.scatter(class1_data[i, 0], class1_data[i, 1], class1_data[i, 2], label = 'rock', c='red', s = 20)
  plt3d.scatter(class2_data[i, 0], class2_data[i, 1], class2_data[i, 2], label = 'classical', c='blue', s = 20)

plt.show()