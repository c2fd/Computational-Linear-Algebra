import numpy as np

def arnoldi(A, x1, m):
    n = A.shape[0]
    q = x1/np.linalg.norm(x1,2)
    Q = np.zeros((n,m+1))
    H = np.zeros((m+1,m))
    Q[:,0] = q

    for k in np.arange(m):
        w = A@q

        for j in np.arange(k+1):
            H[j,k] = np.dot(Q[:,j],w)
            w = w - H[j,k] * Q[:,j]

        w_norm = np.linalg.norm(w,2)
        H[k + 1, k] = w_norm
        if np.abs(w_norm) < 1.0e-8:
            return [Q, H]
        q = w/w_norm
        Q[:,k+1] = q
    return [Q, H]


if __name__ == "__main__":
    #A = - np.array([[-2.0, 1., 0.0, 0.0], [1.0, -2.0, 1.0, 0.0], [0.0, 1.0, -2.0, 1.0], [0.0, 0.0, 1.0, -2.0]])
    #x0 = np.array([1, 1, 1, 1])
    A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
    x0 = np.array([6, 25, -11, 15]).T

    print(x0.shape)
    m = 2

    [Q, H] = arnoldi(A,x0,m)
    print('Q: \n',Q)
    print('H: \n',H)
    print('QTQ:',Q.T@Q)