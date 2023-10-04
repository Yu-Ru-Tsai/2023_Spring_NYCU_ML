import os
import numpy as np
from discrete import get_pixvalueProb_discrete
from utilPlot import *
from utils import *

def load():
    # read file
    train_image_file = open(os.path.join('dataset', 'train-images.idx3-ubyte'), 'rb')
    train_label_file = open(os.path.join('dataset', 'train-labels.idx1-ubyte'), 'rb')

    train_image = np.zeros((60000, 28*28), dtype='uint8')
    train_label = np.zeros(60000, dtype='uint8')

    train_image_file.read(16)
    train_label_file.read(8)

    # Want to get train_image: (60000,28*28), train_label: (60000,)
    for i in range(60000):
        for j in range(28*28):
            train_image[i, j] = int.from_bytes(train_image_file.read(1), byteorder='big')
        train_label[i] = int.from_bytes(train_label_file.read(1), byteorder='big')
    train_image = np.asarray(train_image >= 128,dtype='uint8')
    return (train_image, train_label)

def init_lambda(): # lambda: (10,)
    
    re = np.full((10), 1/10)
    return re

def init_P(A,b): # P: (10,784)

    re = np.zeros((10, 784))
    for i in range(10):
        for j in range(784):
            re[i][j] = np.random.rand()/2 + 0.25

    return re

if __name__ == '__main__':

    eps = 1e-2
    A, b = load()
    Lambda = init_lambda()
    P = init_P(A, b) 
    last_diff, diff, count = 1000, 100, 0
    while abs(last_diff - diff) > eps and diff > eps and count < 15:
        # E-step (calculate W)
        W = E_step(A, Lambda, P) # (60000, 10)

        # M-step (update Lambda, P)
        L_new, P_new = M_step(A, W)
        
        last_diff = diff
        diff = np.sum(np.abs(Lambda - L_new)) + np.sum(np.abs(P - P_new))
        Lambda = L_new
        P = P_new
        count += 1


    maxs = np.argmax(W, axis=1) # 60000 x 1
    unique, counts = np.unique(maxs, return_counts=True)
    print(dict(zip(unique, counts)))
    print(f"Lambda: {Lambda.reshape(1,-1)}")

    GT_distribution = get_pixvalueProb_discrete(A, b)
    order = perfect_matching(GT_distribution, P)

    plot(P, order, threshold=0.35)
    confusion_matrix(b, maxs, order)
    print_error_rate(count, b, maxs, order)
