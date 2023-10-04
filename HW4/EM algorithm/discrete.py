import numpy as np


def get_pixvalueProb_discrete(train_x, train_y):
    
    labels = np.zeros(10)
    for label in train_y:
        labels[label] += 1

    distribution = np.zeros((10, 784))
    for i in range(60000):
        c = train_y[i]
        for j in range(784):
            if train_x[i, j] == 1:
                distribution[c, j] += 1

    # normalized
    distribution = distribution / labels.reshape(-1, 1)

    return distribution
