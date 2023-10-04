import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_data(path):
    X = []
    Y = []
    with open(path, 'r') as f:
        for line in f.readlines():
            x, y = line.split(' ')
            X.append(float(x))
            Y.append(float(y))
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y

def kernel(X1, X2, sigma, alpha, length_scale):
    
    # rational quadratic kernel function: k(x_i, x_j) = sigma*(1 + (x_i-x_j)^2 / (2*alpha * length_scale^2)) ^ -alpha
    kernel = sigma * np.power(1 + np.power(X1.reshape(-1,1) - X2.reshape(1,-1), 2) / (2 * alpha * length_scale ** 2), -alpha)
    return kernel

def predict(x_line, X, y, C, beta, sigma, alpha, length_scale):

    # K(X, X*): 34 X 500
    k_x_xstar = kernel(X, x_line, sigma, alpha, length_scale)

    # K(X*, X*): 500 X 500
    k_xstar_xstar = kernel(x_line, x_line, sigma, alpha, length_scale) 
    
    # mean = k(x,x*)^T C^(-1) y
    means = k_x_xstar.T @ np.linalg.inv(C) @ y.reshape(-1,1) # 500 x 1
    
    # var = k(x*,x*) + 1/beta - k(x,x*) C^(-1) k(x,x*)
    var = k_xstar_xstar + (1/beta) * np.identity(len(k_xstar_xstar)) - k_x_xstar.T @ np.linalg.inv(C) @ k_x_xstar # 500 x 500

    return means, var

def objective_function(theta, X, y, beta):
    
    theta = theta.ravel()
    K = kernel(X, X, sigma=theta[0], alpha=theta[1], length_scale=theta[2]) + (1/beta) * np.identity(len(X))
    L = np.linalg.cholesky(K) # K = LL^T
    nll = 0.5 * y.reshape(1,-1) @ np.linalg.inv(K) @ y.reshape(-1,1) + np.sum(np.log(np.diag(L))) + 0.5 * len(X) * np.log(2*np.pi)
    return nll.item()


if __name__ == '__main__':

    ############### part 1 ################

    # initial value
    X, y = load_data(path='input.data')    # X: 34, y: 34 
    beta = 5
    sigma = 1
    alpha = 1 
    length_scale = 1

    # convariance C(Xn, Xm)
    C = kernel(X, X, sigma, alpha, length_scale) + 1 / beta * np.identity(len(X))  # C(Xn, Xm): 34 x 34

    x_line = np.linspace(-60, 60, num=500)
    mean_predict, variance_predict = predict(x_line, X, y, C, beta, sigma, alpha, length_scale)
    mean_predict = mean_predict.reshape(-1)
    variance_predict = np.sqrt(np.diag(variance_predict))

    # plot
    plt.plot(X, y, 'bo')
    plt.plot(x_line, mean_predict,'k-')

    # 95% confidence interval: +- 2*variance_predict
    plt.fill_between(x_line, mean_predict + 2 * variance_predict, mean_predict - 2 * variance_predict, facecolor='salmon')
    plt.xlim(-60, 60)
    plt.savefig("GP.jpg")
    # plt.show()

    ############### part 2 ################

    opt = minimize(objective_function, x0=[sigma, alpha, length_scale], bounds=((1e-5, 1e5), (1e-5, 1e5), (1e-5, 1e5)), args=(X, y, beta))
    sigma_opt = opt.x[0]
    alpha_opt = opt.x[1]
    length_scale_opt = opt.x[2]
    print(f"sigma_opt: {sigma_opt}")
    print(f"alpha_opt: {alpha_opt}")
    print(f"length_scale_opt: {length_scale_opt}")

    # convariance C(Xn, Xm)
    C = kernel(X, X, sigma=sigma_opt, alpha=alpha_opt, length_scale = length_scale_opt) + 1 / beta * np.identity(len(X))  # C(Xn, Xm): 34 x 34

    x_line = np.linspace(-60, 60, num=500)
    mean_predict, variance_predict = predict(x_line, X, y, C, beta, sigma=sigma_opt, alpha=alpha_opt, length_scale=length_scale_opt)
    mean_predict = mean_predict.reshape(-1)
    variance_predict = np.sqrt(np.diag(variance_predict))

    # plot
    fig = plt.figure()
    plt.plot(X, y, 'bo')
    plt.plot(x_line, mean_predict,'k-')

    # 95% confidence interval: +- 2*variance_predict
    plt.fill_between(x_line, mean_predict + 2 * variance_predict, mean_predict - 2 * variance_predict, facecolor='salmon')
    plt.xlim(-60, 60)
    plt.savefig("opt_GP.jpg")
    # plt.show()
