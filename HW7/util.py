from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

path = "./output/"
H, W = 231, 195

def imread(path, H, W, num_subjects):

    pics = os.listdir(path)
    images = np.zeros((W*H, len(pics)))
    label = np.array([[i]*num_subjects for i in range(15)]).reshape(-1)

    for pic, i in zip(pics, np.arange(len(pics))):
        image = np.asarray(Image.open(os.path.join(path, pic)).resize((W,H), Image.ANTIALIAS)).flatten()
        images[:,i] = image

    return images, label

def show_eigenface(eigenvectors, num, mode, kernel, k):
    
    n = int(num ** 0.5)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    for i in range(num):
        plt.subplot(n, n, i+1)
        plt.imshow(eigenvectors[:,i].reshape(H,W), cmap='gray')
        plt.axis('off')
    os.makedirs(path+kernel, exist_ok=True)
    # plt.savefig(path+kernel+'/'+mode+"_"+"eigenvector"+"_"+str(k)+".png")
    
    # plt.show()

def show_reconstruction(X, X_recover, num, mode, kernel=None, k=None):
    
    randint = np.random.choice(X.shape[1], num)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.01)
    for i in range(num):
        plt.subplot(2, num, i+1)
        plt.imshow(X[:, randint[i]].reshape(H,W), cmap='gray')
        plt.axis('off')
        plt.subplot(2, num, i+1+num)
        plt.imshow(X_recover[:, randint[i]].reshape(H,W), cmap='gray')
        plt.axis('off')
    
    # plt.savefig(path+kernel+'/'+mode+"_"+"reconstruction"+"_"+str(k)+".png")
     
    plt.show()

def performance(X_test, y_test, Z_train, y_train, eigenvectors, X_mean=None, k=5):
    
    if X_mean is None:
        X_mean = np.zeros((X_test.shape[0], 1))

    # reduce dim (projection)
    Z_test = eigenvectors.T @ (X_test - X_mean)

    # k-nn
    predict = np.zeros(Z_test.shape[1])
    for i in range(Z_test.shape[1]):
        distance = np.zeros(Z_train.shape[1])
        for j in range(Z_train.shape[1]):
            distance[j] = np.sum(np.square(Z_test[:,i] - Z_train[:,j]))
        sort_index = np.argsort(distance)

        nearest_neighbors = y_train[np.argsort(distance)[:k]]
        unique, counts = np.unique(nearest_neighbors, return_counts=True)
        predict[i] = unique[np.argmax(counts)]

    acc = np.count_nonzero((y_test - predict) == 0) / len(y_test)

    return acc
