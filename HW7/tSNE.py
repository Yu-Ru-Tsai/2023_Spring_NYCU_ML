#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pylab

file_path = './Experiment Result'
Similarity_path = './Similarity'
output_dir = './gif/'

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X) # Distance [2500, 2500]
    P = np.zeros((n, n))
    beta = np.ones((n, 1)) # beta = 1 / sigma^2
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P

def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

def SNE(X=np.array([]), labels=None, no_dims=2, initial_dims=50, mode='t-SNE', perplexity=30):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real  # 784 -> 50
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims) # Y is goal
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities >> qij
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        if mode == 't-SNE':
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        if mode == 'Symmetric SNE':
            num = np.exp(-1. * np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            if mode == 't-SNE':
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            if mode == 'Symmetric SNE':
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
            
        # Visualize
        if (iter + 1) % 50 == 0:
            visualize(Y, labels, mode, perplexity, (iter + 1))
            
        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y, P, Q

def visualize(Y, labels, mode, perplexity, _iter):

    fig = plt.figure()
    scatter = plt.scatter(Y[:, 0], Y[:, 1], s=20, c=labels)
    plt.legend(*scatter.legend_elements(), loc='lower right', prop={'size': 7.8})
    plt.title(f'{mode} with perplexity={perplexity}, iter={_iter}')
    plt.axis('off')
    fig.savefig(f'{file_path}/{mode}/per_{perplexity}/{_iter}.jpg')

def saveSimilarities(P, Q, mode, perplexity):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'hspace': 0.5})
    ax1.set_title(mode + " high-dim")
    ax1.hist(P.flatten(), bins=40, log=True)

    ax2.set_title(mode + " low-dim")
    ax2.hist(Q.flatten(), bins=40, log=True)

    plt.savefig(f'{Similarity_path}/{mode}/per_{perplexity}_similarity.jpg')

def create_gif(figure_dir, mode, perplexity):

    files = os.listdir(figure_dir)
    files.remove('Final.jpg')
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    frames = []

    for file in files:
        file_path = os.path.join(figure_dir, file)

        image = Image.open(file_path)
        frames.append(image)

    output_file = os.path.join(output_dir, '{}_per{}_optimal procedure'+'.gif').format(mode, perplexity)
    frames[0].save(output_file, format='GIF', append_images=frames[1:], save_all=True, duration=200, loop=0)


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    
    X = np.loadtxt("./MNIST_data/mnist2500_X.txt")  
    labels = np.loadtxt("./MNIST_data/mnist2500_labels.txt")
    
    mode = ['t-SNE', 'Symmetric SNE']  
    perplexity = [5, 20, 35, 50]  
    dims = 2
    init_dims = 50
    for i in mode:
        for j in perplexity:
            figure_dir = '.\\Experiment Result\\{}\\per_{}'.format(i, j)
            try:
                os.mkdir(f'{file_path}/{i}/per_{j}')
            except:
                pass
            Y, P, Q = SNE(X, labels, dims, init_dims, i, j)
            visualize(Y, labels, i, j, 'Final')
            saveSimilarities(P, Q, i, j)
            create_gif(figure_dir, i, j)
        
