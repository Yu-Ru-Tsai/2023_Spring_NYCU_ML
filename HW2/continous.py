import math
import numpy as np

def get_pixvalueProb_continuous(train_image, train_label):
    # pixvalueProb_image
    pixvalueProb_image = np.zeros((10,28*28,256))
    for y in range(10):
        A = train_image[train_label == y]
        for i in range(28*28):
            mu = np.mean(A[:,i])
            var = np.var(A[:,i])
            if var == 0:
                var = 20
            for j in range(256):
                pixvalueProb_image[y,i,j] = gaussain_prob(j, mu, var)

    # pixvalueProb_label
    pixvalueProb_label = np.zeros((10,))
    for i in range(60000):
        c = train_label[i]
        pixvalueProb_label[c] += 1
    pixvalueProb_label[:] /= sum(pixvalueProb_label)
    return pixvalueProb_image, pixvalueProb_label

def gaussain_prob(x, mu, var):
    return ((1/math.sqrt(2*math.pi*var))*math.exp((-(x-mu)**2)/(2*var)))

# pixvalueProb_image: (10,28*28,256), pixvalueProb_label: (10,)
def test_continuous(pixvalueProb_image, pixvalueProb_label, test_image, test_label):
    error = 0
    for i in range(10000):
        probs = np.zeros(10)
        for c in range(10):
            for d in range(28 * 28):  
                probs[c] += np.log(max(1e-30, pixvalueProb_image[c, d, int(test_image[i, d])]))
            probs[c] += np.log(pixvalueProb_label[c])
        # normalized
        probs /= np.sum(probs)
        print(f"Posterior (in log scale):")
        for c in range(10):
            print(f"{c}: {probs[c]}")
        predict = np.argmin(probs)
        print(f"Prediction: {predict}, Ans: {test_label[i]}")
        print()
        if predict != test_label[i]:
            error += 1
    print('Error rate: {:.4f}'.format(error / 10000))
    print()




