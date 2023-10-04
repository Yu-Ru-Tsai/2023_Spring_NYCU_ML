import numpy as np

def matrix_inverse(matrix):
    """計算反矩陣"""
    n = matrix.shape[0]

    # 增廣矩陣(A | I)
    augmented_matrix = np.hstack((matrix, np.eye(n)))

    # 做Gauss-Jordan Elimination得到(I | A得到(I | A^(-1) ) 
    for i in range(n):
        pivot = augmented_matrix[i, i]
        
        # 要避免在pivot=0所以要做列交換
        if pivot == 0:
            for j in range(i+1, n):
                if augmented_matrix[j, i] != 0:
                    temp = augmented_matrix[i, :].copy()
                    augmented_matrix[i, :] = augmented_matrix[j, :]
                    augmented_matrix[j, :] = temp
                    break
            pivot = augmented_matrix[i, i]
            if pivot == 0:raise ValueError("此矩陣不存在反矩陣")
            
        for j in range(2 * n):
            augmented_matrix[i, j] /= pivot
            
        for k in range(n):
            if k == i:
                continue
            factor = augmented_matrix[k, i]
            for j in range(2 * n):
                augmented_matrix[k, j] -= factor * augmented_matrix[i, j]
    inverse = augmented_matrix[:, n:]
    return inverse

if __name__ == '__main__':
    # 測試
    a = np.array([[0,1],[1,0]])    
    print(matrix_inverse(a))
