import numpy as np
import matplotlib.pyplot as plt
from utils import *


def run_gradient(A, w, b, lr=0.01):

    d = 100
    while np.sqrt(np.sum(d**2)) > 1e-2:
        d = A.T@(1/(1+np.exp(-A@w)) - b)
        w = w - lr*d

    return w


def run_newton(A, w, b, lr=0.01):

    N = len(A)
    D = np.zeros((N, N))
    for i in range(N):
        D[i, i] = np.exp(-A[i]@w)/np.power(1+np.exp(-A[i]@w), 2)
    H = A.T@D@A
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError as error:
        print(str(error))
        print('Hessian matrix non invertible, switch to Gradient descent')
        return run_gradient(A, w, b)

    d = 100
    while np.sqrt(np.sum(d**2)) > 1e-2:
        d = H_inv@A.T@(1/(1+np.exp(-A@w))-b)
        w = w - lr*d

    return w


def confusion_matrix(A, b, b_predict):

    # A: 2Nx3, b: 2Nx1, b_predict: 2Nx1

    doubleN = len(A) 
    TP = FP = FN = TN = 0

    for i in range(len(b)):
        if b_predict[i][0] == b[i][0] == 1:
            TP += 1
        elif b_predict[i][0] == b[i][0] == 0:
            TN += 1
        elif b_predict[i][0] == 1 and b[i][0] == 0:
            FP += 1
        else:
            FN += 1

    matrix = np.empty((2, 2))
    matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1] = TP, FN, FP, TN

    C0_predict = []
    C1_predict = []
    for i in range(doubleN):
        if b_predict[i] == 0:
            C0_predict.append(A[i, 1:])
        else:
            C1_predict.append(A[i, 1:])

    return (matrix, np.array(C0_predict), np.array(C1_predict))


def print_confusion_matrix(matrix):

    print(f"Confusion Matrix:")
    print(f"               Predict cluster 1  Predict cluster 2")
    print(f"Is cluster 1        {matrix[0,0]:.0f}                 {matrix[0,1]:.0f}       ")
    print(f"Is cluster 2        {matrix[1,0]:.0f}                  {matrix[1,1]:.0f}       ")
    print()
    print(f"Sensitivity (Successfully predict cluster 1): {matrix[0,0]/(matrix[0,0]+matrix[1,0])}")
    print(f"Specificity (Successfully predict cluster 2): {matrix[0,0]/(matrix[0,0]+matrix[0,1])}")


def visualize(C0, C1, Gd_C0_predict, Gd_C1_predict, Newton_C0_predict, Newton_C1_predict):

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    plot(C0, C1, 'Ground truth', ax1)

    ax2 = fig.add_subplot(1, 3, 2)
    plot(Gd_C0_predict, Gd_C1_predict, 'Gradient descent', ax2)

    ax3 = fig.add_subplot(1, 3, 3)
    plot(Newton_C0_predict,  Newton_C1_predict, 'Newton\'s method', ax3)

    plt.show()


def plot(C0, C1, title, ax):

    ax.plot(C0[:, 0], C0[:, 1], 'ro')
    ax.plot(C1[:, 0], C1[:, 1], 'bo')
    ax.set_title(title)


if __name__ == '__main__':
    N = int(input('N: '))
    mx1, my1 = [int(x) for x in input('mx1與my1: ').split()]
    mx2, my2 = [int(x) for x in input('mx2與my2: ').split()]
    vx1, vy1 = [int(x) for x in input('vx1與vy1: ').split()]
    vx2, vy2 = [int(x) for x in input('vx2與vy2: ').split()]

    C0 = sampling(mx1, my1, vx1, vy1, N)
    C1 = sampling(mx2, my2, vx2, vy2, N)
    A = get_A(N, C0, C1)
    b = get_b(N)

    # gradient descent
    w = np.random.rand(3, 1)
    w = run_gradient(A, w, b, lr=0.001)

    print(f"Gradient descent:")
    print()
    b_predict = predict(A, w)
    ConfusionMatrix, Gd_C0_predict, Gd_C1_predict = confusion_matrix(A, b, b_predict)
    print('w:')
    print(w[0])
    print(w[1])
    print(w[2])
    print()
    print_confusion_matrix(ConfusionMatrix)

    # newton's method
    w = np.random.rand(3, 1)
    w = run_newton(A, w, b, lr=0.001)

    print('\n----------------------------------------')
    print('Newton\s method:')
    print()
    b_predict = predict(A, w)
    ConfusionMatrix, Newton_C0_predict, Newton_C1_predict = confusion_matrix(A, b, b_predict)
    print('w:')
    print(w[0])
    print(w[1])
    print(w[2])
    print()
    print_confusion_matrix(ConfusionMatrix)
    visualize(C0, C1, Gd_C0_predict, Gd_C1_predict, Newton_C0_predict, Newton_C1_predict)
