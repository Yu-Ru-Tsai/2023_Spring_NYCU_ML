import numpy as np
from util import imread, show_eigenface, show_reconstruction, performance

def lda(X, X_mean, X_pca, y, X_test, y_test, eigenvectors_pca, kernel, mode, num_dim=None, k=5):
    
    N = X_pca.shape[0] 
    X_mean_pca = np.mean(X_pca, axis=1).reshape(-1, 1)

    classes_mean = np.zeros((N, 15))  # 15 classes
    for i in range(X_pca.shape[1]):
        classes_mean[:, y[i]] += X_pca[:, i].astype(np.float64)

    classes_mean = classes_mean / 9

    # within-class scatter
    S_within = np.zeros((N, N))
    for i in range(15):
        mask = np.isin(y, [i])
        X_masked = X_pca[:, mask]
        for j in range(X_masked.shape[1]):
            d = X_masked[:, j].reshape(-1, 1) - classes_mean[:, i].reshape(-1, 1)
            S_within += (d @ d.T).astype(np.float64)

    # between-class scatter
    S_between = np.zeros((N, N))
    for i in range(15):
        d = classes_mean[:, i].reshape(-1,1) - X_mean_pca
        S_between += (9 * d @ d.T).astype(np.float64)

    eigenvalues_lda, eigenvectors_lda = np.linalg.eig(np.linalg.inv(S_within) @ S_between)

    sort_index = np.argsort(-eigenvalues_lda)
    if num_dim is None:
        sort_index = sort_index[:-1]  # reduce 1 dim
    else:
        sort_index = sort_index[:num_dim]

    eigenvectors_lda = np.asarray(eigenvectors_lda[:,sort_index].real)

    U = eigenvectors_pca @ eigenvectors_lda 
    show_eigenface(U, 25, mode=mode, kernel=kernel, k=k)
    Z = U.T @ (X - X_mean)

    X_recover = U @ Z + X_mean
    show_reconstruction(X, X_recover, 10, mode=mode, kernel=kernel, k=k)

    # accuracy
    acc = performance(X_test, y_test, Z, y, U, X_mean, k)
    print(f"acc: {acc*100:.2f}%")