import numpy as np
import math
from utils import linear_model_datagen
import matplotlib.pyplot as plt


def plot(num_points, x, mean, variance, title, ax):

    mean_predict = np.zeros(500)
    variance_predict = np.zeros(500)
    for i in range(len(x)):
        X = np.asarray([math.pow(x[i], k) for k in range(n)])
        mean_predict[i] = (X@mean).item()
        variance_predict[i] = ((a) + X@variance@X.T).item()
    ax.plot(point_x[:num_points], point_y[:num_points], 'bo')
    ax.plot(x, mean_predict, 'k-')
    ax.plot(x, mean_predict+variance_predict, 'r-')
    ax.plot(x, mean_predict-variance_predict, 'r-')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-20, 20)
    ax.set_title(title)


def visualize(mean_list, variance_list):

    x = np.linspace(-2, 2, 500)
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    plot(0, x, w, np.zeros((n, n)), 'Ground truth', ax1)

    ax2 = fig.add_subplot(2, 2, 2)
    plot(len(point_x), x, mean_list[2],
         variance_list[2], 'Predict result', ax2)

    ax3 = fig.add_subplot(2, 2, 3)
    plot(10, x, mean_list[0], variance_list[0], 'After 10 incomes', ax3)

    ax4 = fig.add_subplot(2, 2, 4)
    plot(50, x, mean_list[1], variance_list[1], 'After 50 incomes', ax4)
    plt.show()


if __name__ == '__main__':

    b = 1  # prior gaussian distribution's variance
    n = 3  # basis
    a = 3  # variance of N(0,a)
    w = np.asarray([1, 2, 3])  # line parameters

    point_x = []
    point_y = []

    # [after 10 points, after 50 points, final result]
    mean_list = []
    variance_list = []
    m = np.ones((n, 1))  # mean 代換的
    s = np.identity(n)  # var 代換的
    mean = np.zeros((n, 1))
    variance = (1/b)*np.identity(n)
    i = 1
    while (np.average(abs(m - mean)) > 1e-4) or (np.average(abs(s - variance)) > 1e-4) or len(point_x) < 50:
        # add point
        point = linear_model_datagen(n, a, w)
        print(f"Add data point {point}:")
        print()
        # update mean & variance
        X = np.asarray([math.pow(point[0], i)
                       for i in range(n)]).reshape(1, -1)  # 1x4
        y = point[1]
        S = np.linalg.pinv(variance)  # Pseudoinverse
        variance_new = np.linalg.pinv((1/a)*X.T@X + S)
        mean_new = variance_new@((1/a)*X.T*y + S@mean)

        print('Posterior mean:')
        print(mean_new)
        print()
        print('Posterior variance:')
        print(variance_new)
        print()

        # predictive distribution
        predictive_mean = (X@mean_new).item()
        predictive_variance = ((a)+X@variance_new@X.T).item()
        print(
            f"Predictive distribution ~ N({predictive_mean:.5f},{predictive_variance:.5f})")
        print()

        # save record
        point_x.append(point[0])
        point_y.append(point[1])
        if i == 10 or i == 50:
            mean_list.append(mean_new)
            variance_list.append(variance_new)
        m = mean
        s = variance
        mean = mean_new
        variance = variance_new
        i += 1
    mean_list.append(mean_new)
    variance_list.append(variance_new)
    print(f"point: {len(point_x)}")
    visualize(mean_list, variance_list)
