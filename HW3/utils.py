import numpy as np
import math
import random

# Univariate gaussian data generator
def gaussian_datagen(m, s):

    # Marsaglia polar method
    S, U, V = 0, 0, 0
    while True:
        U = np.random.uniform(-1, 1)
        V = np.random.uniform(-1, 1)
        S = U**2 + V**2
        if S < 1:
            break
    x = U * math.sqrt(-2 * math.log(S) / S) # N(0, 1)
    sample = x * (s**0.5) + m # N(m, s)
    return sample

# Polynomial basis linear model data generator
def linear_model_datagen(n, a, w):

    if n != len(w):
        print("Error: input size error.")
        return -1
    x0 = random.uniform(-1,1)
    x = [math.pow(x0,i) for i in range(len(w))]
    y = np.sum(w*x)
    e = gaussian_datagen(0, a)
    return x0, y+e
