import numpy as np

def sd(A,b, x0, tol=1.0e-8, maxit=10):
    m, n = A.shape
    assert (m == n)
    assert (np.all(np.linalg.eigvals(A) > 0))
    assert np.all(A.T == A)

    x = np.array(x0)
    g = np.matmul(A,x) - b
    it = 0
    while np.linalg.norm(g,2) > tol and it < maxit:
        mu = np.dot(g,g)/ np.dot(g, A@g)
        x = x - mu * g
        g = A@x-b
        it = it + 1
    return [x, it]


if __name__ == "__main__":
    A = np.array([[10, -1, 0], [-1, 10, -2], [0, -2, 10]])
    b = np.array([9, 7, 8]).T
    x0 = np.zeros(b.shape)
    tol = 1.0e-8
    maxit = 1000

    [sol, it_num] = sd(A,b,x0,tol,maxit)
    print('x: \n',sol)
    print(it_num)