import numpy as np

def gd(A, b, x0, tol = 1.0e-8):
    x = np.array(x0)
    g = A@x - b
    it = 0

    while np.linalg.norm(g,2) > tol:
        mu = np.dot(g,g)/np.dot(g,A@g)
        x = x - mu * g
        g = A@x - b
        it = it + 1
    return [x, it]

if __name__ == "__main__":
    A = np.array([[10, -1, 0], [-1, 10, -2], [0, -2, 10]])
    b = np.array([9, 7, 8]).T
    x0 = 0.1*np.ones(b.shape)
    tol = 1.0e-8

    [sol, it_num] = gd(A,b,x0,tol)
    print('x: \n',sol)
    print(it_num)

