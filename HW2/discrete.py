import numpy  as np

def get_pixvalueProb_discrete(train_image, train_label):
   
    # train: image: (60000,784) label: (60000)
    # test: image: (10000,784), label: (10000) 
    pixvalueProb_image = np.zeros((10,28*28,32))
    pixvalueProb_label = np.zeros((10,))
    for i in range(60000):
        c = train_label[i]
        pixvalueProb_label[c] += 1
        for j in range(28*28):
            pixvalueProb_image[c][j][int(train_image[i,j])//8] += 1

    count = sum(pixvalueProb_label)
    pixvalueProb_label /= count

    for i in range(10):
        for j in range(28*28):
            count = sum(pixvalueProb_image[i][j][:])
            pixvalueProb_image[i][j][:] /= count

    return pixvalueProb_image, pixvalueProb_label

def test_discrete(pixvalueProb_image, pixvalueProb_label, test_image, test_label): 
    error = 0
    for i in range(10000):
        probs = np.zeros(10)
        for c in range(10):
            for d in range(28 * 28):  
                    probs[c] += np.log(max(1e-6, pixvalueProb_image[c, d, int(test_image[i, d])//8]))
            probs[c] += np.log(pixvalueProb_label[c])
           
        probs /= np.sum(probs)
        print('Posterior (in log scale):')
        for c in range(10):
            print(f"{c}: {probs[c]}")
        predict = np.argmin(probs)
        print(f"Prediction: {predict}, Ans: {test_label[i]}")
        print()
        if predict != test_label[i]:
            error += 1
    print('Error rate: {:.4f}'.format(error / 10000))
    print()


if __name__=='__main__':

    # 測試
    train_image, train_label, test_image, test_label = load()
    pixvalueProb_image, pixvalueProb_label = get_pixvalueProb_discrete(train_image, train_label)
    test_discrete(pixvalueProb_image, pixvalueProb_label, test_image, test_label)
    print(f"pixvalueProb_image={pixvalueProb_label}")