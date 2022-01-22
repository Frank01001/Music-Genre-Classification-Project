import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

genre_names = ['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Dataset import
data = pd.read_csv('./extracted_dataset.csv')

# Dataset normalization
data_mean = data.mean()
data_std = data.std()

data_normalized = (data - data_mean) / data_std

dataset = data_normalized.to_numpy()[:, 1:4]
labels = data_normalized.to_numpy()[:, 4]

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

def phi(x):
  f1 = x[0]
  f2 = x[1]
  f3 = x[2]
  return np.array([f1, f2, f3, f1**2, f2**2, f3**2, f1*f2, f2*f3, f1*f3])


feature_size = 9
w = np.zeros(feature_size)
b = 0

for epoch in range(10000):
    gradient_w = np.zeros(w.size)
    gradient_b = 0
    for sample_index in range(100):
        phi_1 = phi(class1_data[sample_index, :])
        phi_2 = phi(class2_data[sample_index, :])
        gradient_w += hinge_derivative_w(phi_1, 1)
        gradient_w += hinge_derivative_w(phi_2, -1)
        gradient_b += hinge_derivative_b(phi_1, 1)
        gradient_b += hinge_derivative_b(phi_2, -1)
    gradient_w /= 200.0
    gradient_b /= 200.0

    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

print(w)
print(b)

from mpl_toolkits.mplot3d import axes3d

def plot_implicit(fn, bbox=(-2.5,2.5)):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    fig = plt.figure(figsize=(20,12))
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(xmin, xmax, 15) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y')

    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x')

    for i in range(100):
        ax.scatter(class1_data[i, 0], class1_data[i, 1], class1_data[i, 2], label='rock', c='red', s=20)
        ax.scatter(class2_data[i, 0], class2_data[i, 1], class2_data[i, 2], label='classical', c='blue', s=20)

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin, zmax)
    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)

    plt.show()

def surface(x,y,z):
    return w[0]*x+w[1]*y+w[2]*z+w[3]*x**2+w[4]*y**2+w[5]*z**2+w[6]*np.multiply(x,y)+w[7]*np.multiply(y,z)+w[8]*np.multiply(x,z) - b

plot_implicit(surface)