import numpy as np
import random
import math


def gaussian_distribution(m, s):

    # Marsaglia polar method
    S, U, V = 0, 0, 0
    while True:
        U = np.random.uniform(-1, 1)
        V = np.random.uniform(-1, 1)
        S = U**2 + V**2
        if S < 1:
            break
    x = U * math.sqrt(-2 * math.log(S) / S)  # N(0, 1)
    sample = x * (s**0.5) + m  # N(m, s)
    return sample


def sampling(mx, my, vx, vy, N):

    re = np.empty((N, 2))
    for i in range(N):
        re[i, 0] = gaussian_distribution(mx, vx)
        re[i, 1] = gaussian_distribution(my, vy)
    return re


def predict(A, w):
    
    N = len(A)
    b_predict = np.empty((N, 1))
    for i in range(N):
        if A[i]@w < 0:
            b_predict[i] = 0
        else:
            b_predict[i] = 1

    return b_predict


def get_A(N, C0, C1):
    
    A = np.zeros((2 * N, 3))
    A[:, 0] = 1
    A[:, 1:] = np.vstack((C0, C1))
    return A


def get_b(N):
    b = np.zeros((2 * N, 1))
    b[N:] = np.ones((N, 1))
    return b
