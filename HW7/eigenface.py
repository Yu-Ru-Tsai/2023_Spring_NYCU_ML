import numpy as np
import os
import matplotlib.pyplot as plt
from util import imread, show_eigenface, show_reconstruction, performance
from pca import kernel_pca, pca
from lda import lda

if __name__=='__main__':

    H, W = 231, 195
    filepath = os.path.join('Yale_Face_Database', 'Training')
    X, y = imread(filepath, H, W, 9) # X: (45045, 135), y: (135, )
    filepath = os.path.join('Yale_Face_Database', 'Testing')
    X_test, y_test = imread(filepath, H, W, 2)
    k = 1
    # PCA
    print(f"K: {k}")
    eigenvectors_pca, z_train, X_mean = pca(X, y, X_test, y_test, num_dim=None, k=k)
    lda(X, X_mean, z_train, y, X_test, y_test, eigenvectors_pca, 'none', mode='lda', num_dim=None, k=k)

    # Kernel PCA
    kernel = ['linear', 'poly']
    for i in kernel:
        print(f"kernel: {i}")
        eigenvectors_pca, z_train, X_mean = kernel_pca(X, y, X_test, y_test, i, num_dim=None, k=k)
        lda(X, X_mean, z_train, y, X_test, y_test, eigenvectors_pca, i, mode='kernel lda', num_dim=None, k=k)

    