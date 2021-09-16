import numpy as np


def cg(A, b, x0, tol=1.0e-8, maxit=1000):
    m, n = A.shape
    assert (m == n)
    assert (np.all(np.linalg.eigvals(A) > 0))
    assert np.all(A.T == A)

    r = b - A@x0
    p = np.array(r)
    x = np.array(x0)

    x_old = np.array(x)
    r_old = np.array(r)
    p_old = np.array(p)

    for it in np.arange(maxit):
        alpha = np.dot(r_old, r_old) / np.dot(p_old, A@p_old)
        x = x_old + alpha * p_old
        r = r_old - alpha * A@p_old
        if np.linalg.norm(r, 2) < tol:
            iter_num = it
            return [x, iter_num]

        beta = np.dot(r,r)/np.dot(r_old, r_old)
        p = r + beta * p_old

        x_old = x
        r_old = r
        p_old = p
    iter_num = -1
    return [x, iter_num]


if __name__ == "__main__":
    A = np.array([[10, -1, 0], [-1, 10, -2], [0, -2, 10]])
    b = np.array([9, 7, 8]).T
    tol = 1.0e-8
    maxit = 1000
    x0 = np.zeros(b.shape)

    [sol, it_num] = cg(A, b, x0, tol, maxit)
    assert(it_num > 0)
    print('x: \n', sol)
    print(it_num)