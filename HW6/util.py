import numpy as np
import random
from PIL import Image
import os
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def load_png(input):
    img = Image.open('data/' + input)
    img = np.array(img.getdata())  # (10000,3)
    return img


def RBF_kernel(X, gamma_s, gamma_c):  # img, spatial, color

    dist_c = cdist(X, X, 'sqeuclidean')  # (10000, 10000)
    X_spatial = np.array([[i, j] for i in range(100) for j in range(100)])  # (10000, 2)
    dist_s = cdist(X_spatial, X_spatial, 'sqeuclidean')
    RBF_s = np.exp(-gamma_s * dist_s)    # (10000,10000)
    RBF_c = np.exp(-gamma_c * dist_c)    # (10000,10000)
    kernel = np.multiply(RBF_s, RBF_c)   # (10000,10000)

    return kernel


def initial_center(k, kernel, mode="kmeans++"):

    if mode == "random":
        centers_idx = list(random.sample(range(0, 10000), k))

    elif mode == "kmeans++":
        centers_idx = random.sample(range(kernel.shape[0]), 1)
        found = 1
        while found < k:

            dist = np.zeros(kernel.shape[0])

            for i in range(kernel.shape[0]):
                min_dist = np.Inf

                for f in range(found):
                    tmp = np.linalg.norm(kernel[i] - kernel[centers_idx[f]])

                    if tmp < min_dist:
                        min_dist = tmp

                dist[i] = min_dist

            dist = dist / np.sum(dist)
            idx = np.random.choice(np.arange(kernel.shape[0]), size=1, p=dist)
            centers_idx.append(idx[0])
            found += 1

    return centers_idx


def kmeans(k, kernel, output_dir, mode='kmeans++'):

    iter = 0
    Mean = np.zeros((k, kernel.shape[1]))
    centers_idx = initial_center(k, kernel, mode)

    for i in range(len(centers_idx)):
        Mean[i] = kernel[centers_idx[i]].real

    cluster = np.zeros(len(kernel), dtype=np.uint8)
    diff = 1e9
    count = 1
    while diff > 1e-12:
        # E-step
        for i in range(len(kernel)):
            dist = []
            for j in range(k):
                dist.append(np.sqrt(np.sum((kernel[i] - Mean[j])**2)))
            cluster[i] = np.argmin(dist)

        # M-step
        New_Mean = np.zeros(Mean.shape, dtype=np.float64)
        for i in range(k):
            belong = np.argwhere(cluster == i).reshape(-1)
            for j in belong:
                New_Mean[i] += kernel[j].real
            if len(belong) > 0:
                New_Mean[i] = New_Mean[i] / len(belong)
        diff = np.linalg.norm(Mean - New_Mean)
        Mean = New_Mean
        save_png(k, cluster, iter, output_dir)
        iter += 1

    return cluster


def save_png(K, cluster, iter, output_dir):

    colors = np.array([[175,208,201], [145,161,186], [81,98,142], [24,32,68], [14,18,45]])
    result = np.zeros((100*100, 3))

    for k in range(K):
        mask = np.isin(cluster, [k])
        result[mask] = colors[cluster[mask]]
    
    img = result.reshape(100, 100, 3)
    img = Image.fromarray(np.uint8(img))
    img.save(os.path.join(output_dir, '%06d.png' % iter))


def create_gif(figure_dir, output_dir, filename):

    files = sorted(os.listdir(figure_dir))
    frames = []

    for file in files:
        file_path = os.path.join(figure_dir, file)

        image = Image.open(file_path)
        frames.append(image)

    output_file = os.path.join(output_dir, filename+'.gif')
    frames[0].save(output_file, format='GIF',
                   append_images=frames[1:], save_all=True, duration=200, loop=0)
