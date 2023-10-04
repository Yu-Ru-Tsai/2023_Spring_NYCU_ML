import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='image1.png')
parser.add_argument('--input_name', type=str, default='image1')
parser.add_argument('--mode', type=int, default=2, help="1:unnormalized spectral, 2:normalized spectral")
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--s', type=float, default=0.00001)
parser.add_argument('--c', type=float, default=0.0001)
parser.add_argument('--mode1', type=str, default='kmeans++')
args = parser.parse_args()


def normalize_rows(A):

    row, col = A.shape
    A1 = abs(A)
    sigma = np.sum(A1, axis=1)
    for r in range(row):
        A[r, :] /= sigma[r]

    return A


def D_minus_half_square_root(D):
    Dsym = np.zeros((D.shape))
    for i in range(len(D)):
        if D[i,i] != 0:
            Dsym[i,i] = D[i,i]**-0.5
    
    return Dsym


def show_eigen2D(K, data, cluster):
    colors = ['b', 'r']
    plt.clf()
    data = data.real
    for k in range(K):
        mask = np.isin(cluster, [k]) 
        plt.scatter(data[mask,0], data[mask,1], c=colors[k], alpha=0.1)
    plt.savefig("eigen_" + args.input_name + '_' + str(args.k)+ '_' + str(args.mode) + ".png")


def show_eigen3D(K, data, cluster, path):
    colors = ['b', 'r', 'g']
    plt.clf()
    data = data.real
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for k in range(K):
        mask = np.isin(cluster, [k]) 
        ax.scatter(data[mask,0], data[mask,1], data[mask,2], c=colors[k], alpha=0.1)

    plt.savefig(path+"eigen_" + args.input_name + '_' + str(args.k)+ '_' + str(args.mode) + ".png")


if __name__ == '__main__':

    if args.mode == 1:  # unnormalized
        print("ratio spectral")
        output_dir = os.path.join('spectral_clustering_unnormalized', args.mode1, 'k_{}'.format(args.k), args.input_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        X_color = load_png(args.input)  # (10000, 3)

        ################ laplacian ##################

        # W = RBF_kernel(X_color, args.s, args.c)
        # D = np.diag(np.sum(W, axis=1))
        # L = D - W
        
        # ****************** eigenvalue ******************
        
        # eigenvalue, eigenvector = np.linalg.eig(L) # (100,), (100, 100)
        # np.save(f"unnormalized_eigenvalue_{args.input_name}.npy", eigenvalue)
        # np.save(f"unnormalized_eigenvector.npy_{args.input_name}", eigenvector) 
        
        eigenvalue = np.load(f"unnormalized_eigenvalue_{args.input_name}.npy")
        eigenvector = np.load(f"unnormalized_eigenvector_{args.input_name}.npy")

        # ****************** eigenvalue ******************

        sort_idx = np.argsort(eigenvalue)
        mask = eigenvalue[sort_idx] > 0
        sort_idx = sort_idx[mask]
        U = eigenvector[:, sort_idx[0:args.k]]
        cluster = kmeans(args.k, U, output_dir, args.mode1)

        # eigenspace
        eigen_path = os.path.join("spectral_clustering_normalized\\normalized_eigenspace\\")
        if args.mode1 == 'kmeans++':
            if args.k == 2:
                show_eigen2D(args.k, U, cluster, eigen_path)
            elif args.k == 3:
                show_eigen3D(args.k, U, cluster, eigen_path)

        # GIF 
        GIF_dir = os.path.join('spectral_clustering_unnormalized_GIF', args.mode1, 'k_{}'.format(args.k)) 
        if not os.path.exists(GIF_dir):
            os.mkdir(GIF_dir)
        create_gif(output_dir, GIF_dir, args.input_name)

    elif args.mode == 2:
        print("Normalized Spectral Clustering")
        output_dir = os.path.join('spectral_clustering_normalized', args.mode1, 'k_{}'.format(args.k), args.input_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        X_color = load_png(args.input)  # (10000, 3)

        ################ laplacian ##################

        # W = RBF_kernel(X_color, args.s, args.c)
        # D = np.diag(np.sum(W, axis=1))
        # L = D - W
        # Dsym = D_minus_half_square_root(D)
        # Lsym = Dsym.dot(L).dot(Dsym)
        
        # ****************** eigenvalue ******************

        # eigenvalue, eigenvector = np.linalg.eig(Lsym)
        # np.save(f"normalized_eigenvalue_{args.input_name}.npy", eigenvalue)
        # np.save(f"normalized_eigenvector_{args.input_name}.npy", eigenvector)
        eigenvalue = np.load(f"normalized_eigenvalue_{args.input_name}.npy")
        eigenvector = np.load(f"normalized_eigenvector_{args.input_name}.npy")
        
        # ****************** eigenvalue ******************

        sort_idx = np.argsort(eigenvalue)
        mask = eigenvalue[sort_idx] > 0
        sort_idx = sort_idx[mask]
        U = eigenvector[:, sort_idx[0:args.k]]
        T = normalize_rows(U)
        cluster = kmeans(args.k, T, output_dir, args.mode1)

        # eigenspace 
        eigen_path = os.path.join("spectral_clustering_normalized\\normalized_eigenspace\\")
        if args.mode1 == 'kmeans++':
            if args.k == 2:
                show_eigen2D(args.k, U, cluster, eigen_path)

            elif args.k == 3:
                show_eigen3D(args.k, U, cluster, eigen_path)

        # GIF 
        GIF_dir = os.path.join('spectral_clustering_normalized_GIF', args.mode1, 'k_{}'.format(args.k)) 
        if not os.path.exists(GIF_dir):
            os.mkdir(GIF_dir)
        create_gif(output_dir, GIF_dir, args.input_name)
