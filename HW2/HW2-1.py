import numpy as np
import matplotlib.pyplot as plt
import os
from discrete import *
from continous import *


def load():
    # read file
    train_image_file = open(os.path.join('dataset', 'train-images.idx3-ubyte'), 'rb')
    train_label_file = open(os.path.join('dataset', 'train-labels.idx1-ubyte'), 'rb')
    test_image_file = open(os.path.join('dataset', 't10k-images.idx3-ubyte'), 'rb')
    test_label_file = open(os.path.join('dataset', 't10k-labels.idx1-ubyte'), 'rb')

    train_image = np.zeros((60000, 28*28), dtype='uint8')
    train_label = np.zeros(60000, dtype='uint8')
    test_image = np.zeros((10000, 28*28), dtype='uint8')
    test_label = np.zeros(10000, dtype='uint8')

    train_image_file.read(16)
    train_label_file.read(8)
    test_image_file.read(16)
    test_label_file.read(8)

    # Want to get train_image: (60000,28*28), train_label: (60000,), test_image: (10000,28*28), test_label: (10000,)
    for i in range(60000):
        for j in range(28*28):
            train_image[i, j] = int.from_bytes(train_image_file.read(1), byteorder='big')
        train_label[i] = int.from_bytes(train_label_file.read(1), byteorder='big')

    for i in range(10000):
        for j in range(28*28):
            test_image[i, j] = int.from_bytes(test_image_file.read(1), byteorder='big')
        test_label[i] = int.from_bytes(test_label_file.read(1), byteorder='big')

    return train_image, train_label, test_image, test_label


def print_predImg_numbers(pixvalueProb_image, threshold):
    print('Imagination of numbers in Bayesian classifier:')
    for n in range(10):
        print(f"{n}:")
        for i in range(28):
            for j in range(28):
                pre = np.argmax(pixvalueProb_image[n, i*28 + j, :])
                print("1" if pre > threshold else "0", end=' ')
            print()
        print()
    print()


if __name__ == '__main__':

    train_image, train_label, test_image, test_label = load()
    # plt.imshow(train_image[1,:].reshape(28,28))
    # plt.show()
    mode = int(input('Toggle option (0: discrete mode / 1: continuous mode): '))
    if mode == 0:
        pixvalueProb_image, pixvalueProb_label = get_pixvalueProb_discrete(train_image, train_label)
        # test_discrete(pixvalueProb_image, pixvalueProb_label, test_image, test_label)
        print_predImg_numbers(pixvalueProb_image, 16)
    else:
        pixvalueProb_image, pixvalueProb_label = get_pixvalueProb_continuous(train_image, train_label)
        test_continuous(pixvalueProb_image, pixvalueProb_label, test_image, test_label)
        print_predImg_numbers(pixvalueProb_image, 128)
