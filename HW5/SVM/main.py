from libsvm.svmutil import *
import numpy as np
import csv

def load(path, type):
    re = []
    if type == 'image':
        with open(path) as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                re.append([float(v) for v in row])
        re = np.asarray(re, dtype='float')

    if type == 'label':
        with open(path) as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
               re.append(int(row[0]))
        re = np.asarray(re, dtype=int)
    return re


def linearKernel(X1, X2):
    kernel = X1 @ X2.T
    return kernel

def RBFKernel(X1, X2, gamma):
    dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
    kernel = np.exp((-1 * gamma * dist))
    return kernel


def mixture_kernel(X1, X2, gamma):
    kernel_linear = linearKernel(X1, X2)
    kernel_RBF = RBFKernel(X1, X2, gamma)
    kernel = kernel_linear + kernel_RBF
    kernel = np.hstack((np.arange(1, len(X1) + 1).reshape(-1,1), kernel))
    return kernel


def GridSearch(kernel_type, X_train, y_train, X_test, y_test):
    costs = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10000]
    gammas = [1/784, 1, 1e-1, 1e-2, 1e-3, 1e-4]
    max_acc = 0.0

    if kernel_type == 'Linear':
        for cost in costs:
            param = "-t 0 -c {} -v 3 -q".format(cost)
            ACC = svm_train(y_train, X_train, param)
            if ACC > max_acc:
                max_acc = ACC
                max_param = param
        
        param_list = max_param.split()
        if "-v" in param_list:
            idx = param_list.index("-v")
            param_list.pop(idx)
            param_list.pop(idx)
        max_param = " ".join(param_list)    
        model = svm_train(y_train, X_train, max_param)
        _, accuracy, _ = svm_predict(y_test, X_test, model)

    elif kernel_type == 'Polynomial':
        for cost in costs:
            param = '-q -t 1 -v 3 -c {}'.format(cost)
            ACC = svm_train(y_train, X_train, param)
            if ACC > max_acc:
                max_acc = ACC
                max_param = param

        param_list = max_param.split()
        if "-v" in param_list:
            idx = param_list.index("-v")
            param_list.pop(idx)
            param_list.pop(idx)
        max_param = " ".join(param_list)    
        model = svm_train(y_train, X_train, max_param)
        _, accuracy, _ = svm_predict(y_test, X_test, model)

    elif kernel_type == 'RBF':
        for cost in costs:
            for gamma in gammas:
                param = '-q -t 2 -v 3 -c {} -g {}'.format(cost, gamma)
                ACC = svm_train(y_train, X_train, param)
                if ACC > max_acc:
                    max_acc = ACC
                    max_param = param

        param_list = max_param.split()
        if "-v" in param_list:
            idx = param_list.index("-v")
            param_list.pop(idx)
            param_list.pop(idx)
        max_param = " ".join(param_list)    
        model = svm_train(y_train, X_train, max_param)
        _, accuracy, _ = svm_predict(y_test, X_test, model)

    
    print(f"kernel_type: {kernel_type}")
    print(f"Accuracy: {accuracy[0]: .2f}%")
    print(f"Best_param: {max_param}")

if __name__=='__main__':

    ############## Part 1 ##############

    X_train = load('X_train.csv', 'image')  # (5000,784)
    Y_train = load('Y_train.csv', 'label')  # (5000, )
    X_test = load('X_test.csv', 'image')    # (2500,784)
    Y_test = load('Y_test.csv', 'label')    # (2500, )
    kernel_param = ['-t 0', '-t 1', '-t 2'] # linear, polynomial, RBF
    accuracy = []
    
    for param in kernel_param:
        model = svm_train(Y_train, X_train, '-q ' + param)
        p_label, p_acc, p_vals = svm_predict(Y_test, X_test, model, '-q') 
        accuracy.append(p_acc[0]) # p_acc: (accuracy, mse, scc)

    print(f"linear kernel accuracy: {accuracy[0]: .2f}%")
    print(f"polynomial kernel accuracy: {accuracy[1]: .2f}%")
    print(f"radial basis function kernel accuracy: {accuracy[2]: .2f}%")

    ############## Part 2 ##############
    print()
    print('Part 2')
    GridSearch('Linear', X_train, Y_train, X_test, Y_test)
    GridSearch('Polynomial', X_train, Y_train, X_test, Y_test)
    GridSearch('RBF', X_train, Y_train, X_test, Y_test)

    ############## Part 3 ##############
    print()
    print('Part 3')
    kernel_train = mixture_kernel(X_train, X_train, 1/784)
    prob = svm_problem(Y_train, kernel_train, isKernel = True)
    param = svm_parameter('-q -t 4')
    model = svm_train(prob, param)

    kernel_test = mixture_kernel(X_test, X_train, 1/784)
    p_label, p_acc, p_vals = svm_predict(Y_test, kernel_test, model, '-q')
    print(f"linear kernel + RBF kernel accuracy: {p_acc[0]: .2f}%")

