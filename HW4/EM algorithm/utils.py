import numpy as np
from scipy.optimize import linear_sum_assignment
from numpy import matmul as mul

# e-step
def E_step(X_train, Lambda, P):

    P_complement = 1 - P
    W = np.zeros((60000,10))
    for i in range(60000):
        for j in range(10):
            W[i][j] = Lambda[j]
            W[i][j] *= np.prod(P[j] ** X_train[i] )
            W[i][j] *= np.prod((P_complement[j]) ** (1 - X_train[i]))

    # normalized each row to [0,1] & sum=1
    sums = np.sum(W, axis=1).reshape(-1,1) # W: 60000 x 10 , sums: 60000 x 1
    W = W/sums

    return W

# m-step
def M_step(A, W):

    Lambda = np.zeros((10,))
    P = np.zeros((10, 784))
    for n in range(10):

        # Update lambda
        Lambda[n] = sum(W[:, n]) / 60000
        wn = W[:, n]
        for i in range(28*28):
            xd = A[:, i]
            P[n][i] = mul(wn.T, xd) / sum(W[:, n])
            P[n][i] = 1e-5 if P[n][i] == 0 else P[n][i]    
    
    return Lambda, P 

def perfect_matching(ground_truth, estimate):

    Cost = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            Cost[i,j] = np.linalg.norm(ground_truth[i] - estimate[j]) # distance

    row_idx, col_idx = linear_sum_assignment(Cost) # hungarian_algorithm
    return col_idx
