import numpy as np


def arnoldi(A, x0, m):
    n = A.shape[0]
    q = x0/np.linalg.norm(x0,2)
    H = np.zeros((m+1,m))
    Q = np.zeros((n,m+1))
    Q[:, 0] = q

    for it1 in np.arange(m):
        w = A@q
        for it2 in np.arange(it1+1):
            H[it2, it1] = np.dot(Q[:, it2], w)
            w = w - H[it2, it1] * Q[:, it2]

        w_norm = np.linalg.norm(w,2)
        H[it1 + 1, it1] = w_norm
        if np.abs(w_norm) < 1.0e-10:
            print('Done!')
            return [Q, H]
        q = w/w_norm
        Q[:, it1 + 1] = q
    return [Q, H]


if __name__ == "__main__":
    #A = - np.array([[-2.0, 1., 0.0, 0.0], [1.0, -2.0, 1.0, 0.0], [0.0, 1.0, -2.0, 1.0], [0.0, 0.0, 1.0, -2.0]])
    #x0 = np.array([1, 1, 1, 1])
    A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
    x0 = np.array([6, 25, -11, 15])

    print(x0.shape)
    m = 2

    [Q, H] = arnoldi(A,x0,m)
    print('Q: \n',Q)
    print('H: \n',H)
    print('QTQ:',Q.T@Q)