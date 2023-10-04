import numpy as np
import matplotlib.pyplot as plt
from utils import gaussian_datagen, linear_model_datagen

if __name__ == '__main__':

    m = 20  
    s = 10    
    samples = [gaussian_datagen(m, s) for i in range(1000)]
    plt.hist(samples, 50)
    plt.title(f"mean:{m}, varinance:{s}")
    plt.show()

    n = 3  #basis number
    a = 10 #variance of N(0,a)
    w = np.asarray([2,5,4])
    x0, y = linear_model_datagen(n, a, w)
    print(f"({x0}, {y})")