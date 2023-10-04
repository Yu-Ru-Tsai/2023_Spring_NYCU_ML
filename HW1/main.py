import numpy as np
import matplotlib.pyplot as plt
import os
from metric import rLSE, newton_method


def show_fitting_line(parameters):
    parameters = parameters.reshape(-1)
    n = len(parameters)-1
    print('Fitting line: ', end='')
    # x^(n-1) ~ x^1
    for i in range(n, 0, -1):
        print(parameters[i], 'X^', i, ' + ', end='')
    # x^0
    print(parameters[0])


def plot(data_x, data_y, parameters_rlse, parameters_newton):
    # rLSE
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.title('rLSE')
    plt.plot(data_x, data_y, 'ro')
    x = np.linspace(min(data_x)-1, max(data_x)+1, 100)
    y = np.zeros(x.shape)
    for i in range(len(parameters_rlse)):
        y += parameters_rlse[i] * np.power(x, i)
    plt.plot(x, y, '-k')

    # Newton
    plt.subplot(2, 1, 2)
    plt.title('Newton\'s Method')
    plt.plot(data_x, data_y, 'ro')
    y = np.zeros(x.shape)
    for i in range(len(parameters_newton)):
        y += parameters_newton[i] * np.power(x, i)
    plt.plot(x, y, '-k')
    plt.savefig("visualization.jpg")
    plt.show()


if __name__ == '__main__':
    data_x = []
    data_y = []

    filepath = os.path.join('testfile.txt')
    with open(filepath) as fp:
        for line in fp:
            a, b = line.split(',')
            data_x.append(float(a))
            data_y.append(float(b))

    data_x = np.asarray(data_x, dtype='float').reshape((-1, 1))  # 23 x 1
    data_y = np.asarray(data_y, dtype='float').reshape((-1, 1))  # 23 x 1

    # Solve linear system
    while True:
        polynomial_basis_size = int(input('n: '))
        Lambda = int(input('lambda: '))
        # A : design matrix
        A = np.empty((len(data_x), polynomial_basis_size))
        for j in range(polynomial_basis_size):
            A[:, j] = np.power(data_x, j).reshape(-1)

        # rLSE
        parameters_rlse, loss_rlse = rLSE(A, Lambda, data_y)
        print('LSE:')
        show_fitting_line(parameters_rlse)
        print(f"Total error: {loss_rlse}")
        print()

        # Netwon's method
        parameters_newton, loss_newton = newton_method(A, data_y)
        print('Newton\'s Method:')
        show_fitting_line(parameters_newton)
        print(f"Total error: {loss_newton}")
        plot(data_x.reshape(-1), data_y.reshape(-1),
             parameters_rlse.reshape(-1), parameters_newton.reshape(-1))
        break
