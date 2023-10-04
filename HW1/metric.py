import numpy as np
from matrix_inverse import matrix_inverse

def rLSE(A, Lambda, b):

    m, n = A.shape
    x = matrix_inverse(A.T@A + Lambda*np.identity(n))@A.T@b
    loss = np.sum(np.square(A.dot(x)-b))
    return x, loss

def newton_method(A, b):

    m, n = A.shape
    x0 = np.random.rand(n,1)
    error = 100
    while error > 1e-7: 
        x1 = x0 - matrix_inverse(2*A.T@A)@(2*A.T@A@x0 - 2*A.T@b)
        error = abs(np.sum(np.square(x1-x0)) / n)
        x0 = x1
    loss = np.sum(np.square(A.dot(x0)-b))
    return x0, loss

