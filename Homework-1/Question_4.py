import numpy as np

#printin without e in the print statements
np.set_printoptions(suppress=True)


#funciton for finding Singular Value decomposition
# Input: Matrix A
# Output: U, Sigma, V
def find_SVD (A):

    #Finding U = A . A_T
    A_AT = np.matmul(A, np.transpose(A))
    Sigma_U, U = np.linalg.eig(A_AT)
    sort_index = Sigma_U.argsort()[::-1]
    Sigma_U = Sigma_U[sort_index]
    U = U[:, sort_index]
    Sigma_U = np.diag(np.sqrt(Sigma_U))
    Sigma_U = np.append(Sigma_U, np.zeros((U.shape[0], 1)), axis=1)
    #print(Sigma_U)
    
    
    #Finding V = A_T . A
    AT_A = np.matmul(np.transpose(A), A)
    Sigma_V, V = np.linalg.eig(AT_A)
    sort_index = Sigma_V.argsort()[::-1]
    Sigma_V = Sigma_V[sort_index]
    V = V[:, sort_index]
    V = np.matmul(np.linalg.pinv(np.matmul(U, Sigma_U)), A)
    
    return U, Sigma_U, V

    
    
    
if __name__ == '__main__':
    
    #given points
    points = np.array([[5, 5, 100, 100], [150, 5, 200, 80], [150, 150, 220, 80], [5, 150, 100, 200]])
    
    x = points[:, 0]
    y = points[:, 1]
    xp = points[:, 2]
    yp = points[:, 3]
    
    #Creating the A matrix
    A = np.empty((8,9))
    j = 0
    for i in range(0, A.shape[0]):
    
        if i%2 != 0:
            A[i] = [0, 0, 0, -x[j], -y[j], -1, (x[j] * yp[j]), (y[j] * yp[j]), yp[j]]
            j += 1
        else:
            A[i] = [-x[j], -y[j], -1, 0, 0, 0, (x[j] * xp[j]), (y[j] * xp[j]),xp[j]]
        
    #Finding U, Sigma and V using SVD
    U, Sigma, V = find_SVD(A)
    V_T = np.transpose(V)
    
    #Printing the computed values
    print('U Matrix is:\n')
    print(U)
    print('\nSigma Matrix is:\n')
    print(Sigma)
    print('\nV Matrix is:\n')
    print(V)
    print('\nComputed A matrix is:\n')
    print(np.round(np.matmul(np.matmul(U, Sigma), V), 4))
    print('\nOriginal matrix:\n')
    print(A)
    print('\nFor a Homogeneous linear equation, x in Ax=0 is: \n')
    print(np.round(V_T[-1], 4))
    print('\nHomography matrix H is :\n')
    print(np.round(V_T[-1].reshape((3, 3)), 4))