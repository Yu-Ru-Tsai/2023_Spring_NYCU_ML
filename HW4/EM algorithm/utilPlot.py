import numpy as np

def plot(GT_Distribution, classes_order, threshold):
    
    Pattern = np.asarray(GT_Distribution > threshold, dtype='uint8')
    for i in range(10):
        print(f"class {i}")
        plot_pattern(Pattern[classes_order[i]])
    return

def plot_pattern(pattern):
    
    for i in range(28):
        for j in range(28):
            print(pattern[i * 28 + j], end=' ')
        print()
    print()
    print()
    return

def confusion_matrix(real, predict, classes_order):
    
    for i in range(10):
        c = classes_order[i]
        TP, FN, FP, TN = 0, 0, 0, 0
        for i in range(60000):
            if real[i] != c and predict[i] != c:
                TN += 1
            elif real[i] == c and predict[i] == c:
                TP += 1
            elif real[i] != c and predict[i] == c:
                FP += 1
            else:
                FN += 1
        plot_confusion_matrix(c, TP, FN, FP, TN)

def plot_confusion_matrix(c, TP, FN, FP, TN):
    print('------------------------------------------------------------')
    print()
    print(f"Confusion Matrix {c}:")
    print(f"                   Predict number {c}     Predict not number {c}")
    print(f"Is number {c}             {TP}                       {FN}")
    print(f"Isn\'t number {c}         {FP}                     {TN}")
    print()
    print(f"Sensitivity (Successfully predict number {c}    ): {TP / (TP + FN):.5f}")
    print(f"Specificity (Successfully predict not number {c}): {TN / (TN + FP):.5f}")
    print()


def print_error_rate(count, real, predict, classes_order):
  
    print(f"Total iteration to converge: {count}")
    real_transform = np.zeros(60000)
    for i in range(60000):
        real_transform[i] = classes_order[real[i]]
    error = np.count_nonzero(real_transform - predict)
    print(f"Total error rate: {error / 60000}")