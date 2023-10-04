import numpy as np
from util import imread, show_eigenface, show_reconstruction, performance

def kernel(X, mode):

    if mode == "linear":
        S = X.T @ X
    elif mode == "poly":
        S = (0.01 * X.T @ X)**3
    N = X.shape[1]
    one_N = np.ones((N, N)) / N
    S = S - one_N @ S - S @ one_N + one_N @ S @ one_N
    return S

def kernel_pca(X, y, X_test, y_test, kernel_type, num_dim=None, k=5):

    X_mean = np.mean(X, axis=1).reshape(-1, 1)
    X_center = X - X_mean
    S = kernel(X, mode=kernel_type)
    eigenvalues, eigenvectors = np.linalg.eig(S)
    sort_index = np.argsort(-eigenvalues)
    if num_dim is None:
        for eigenvalue, i in zip(eigenvalues[sort_index].real, np.arange(len(eigenvalues))):
            if eigenvalue <= 0:
                sort_index = sort_index[:i]
                break
    else:
        sort_index = sort_index[:num_dim]

    eigenvalues = eigenvalues[sort_index]
    eigenvectors = X_center @ eigenvectors[:, sort_index].real
    eigenvectors_norm = np.linalg.norm(eigenvectors, axis=0)
    eigenvectors = eigenvectors / eigenvectors_norm
    show_eigenface(eigenvectors, 25, mode='Kernel PCA', kernel=kernel_type, k=k)

    Z = eigenvectors.T @ X_center 

    X_recover = eigenvectors @ Z + X_mean # X_recover: (45045, 135)
    show_reconstruction(X, X_recover, 10, mode='Kernel PCA', kernel=kernel_type, k=k)
    acc = performance(X_test, y_test, Z, y, eigenvectors, X_mean, k=5)

    print(f"acc: {acc*100:.2f}%")

    return eigenvectors, Z, X_mean

def pca(X, y, X_test, y_test, num_dim=None, k=5):

    X_mean = np.mean(X, axis=1).reshape(-1, 1) 
    X_center = X - X_mean
    
    eigenvalues, eigenvectors = np.linalg.eig(X_center.T @ X_center)
    sort_index = np.argsort(-eigenvalues)
    if num_dim is None:
        for eigenvalue, i in zip(eigenvalues[sort_index], np.arange(len(eigenvalues))):
            if eigenvalue <= 0:
                sort_index = sort_index[:i]
                break
    else:
        sort_index = sort_index[:num_dim]

    eigenvalues = eigenvalues[sort_index]
    eigenvectors = X_center @ eigenvectors[:, sort_index]
    eigenvectors_norm = np.linalg.norm(eigenvectors,axis=0)
    eigenvectors = eigenvectors/eigenvectors_norm

    show_eigenface(eigenvectors, 25, mode='PCA', kernel='none', k=k)
    Z = eigenvectors.T @ (X - X_mean)
    X_recover = eigenvectors @ Z + X_mean

    show_reconstruction(X, X_recover, 10, mode='PCA', kernel='none', k=k)
    acc = performance(X_test, y_test, Z, y, eigenvectors, X_mean, 3)
    print(f"acc: {acc*100:.2f}%")

    return eigenvectors, Z, X_mean