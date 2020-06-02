from mlxtend.data import loadlocal_mnist
import numpy as np

X_train, y_train = loadlocal_mnist(
        images_path='dataset/train-images-idx3-ubyte',
        labels_path='dataset/train-labels-idx1-ubyte')

X_test, y_test = loadlocal_mnist(
        images_path='dataset/t10k-images-idx3-ubyte',
        labels_path='dataset/t10k-labels-idx1-ubyte')

X_train = 0 + (X_train > 127)
X_test = 0 + (X_test > 127)

np.savetxt(fname='newDataset/training-images.csv',
           X=X_train, delimiter=',')
np.savetxt(fname='newDataset/training-labels.csv',
           X=y_train, delimiter=',')
np.savetxt(fname='newDataset/testing-images.csv',
           X=X_test, delimiter=',')
np.savetxt(fname='newDataset/testing-labels.csv',
           X=y_test, delimiter=',')