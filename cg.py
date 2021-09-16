import numpy as np

def cg(A,b,x1, tol = 1.0e-9, numiter = 1000):
    r_i_1 = b - A@x1
    p_i_1 = np.array(r_i_1)  # r_i_1.copy()
    x_i_1 = np.array(x1)
    #r = np.array(r_i_1)

    for i in np.arange(numiter):
        alpha = np.dot(r_i_1,r_i_1)/np.dot(p_i_1, A@p_i_1)
        x_i = x_i_1     + alpha * p_i_1
        r_i = r_i_1 - alpha * A@p_i_1

        if np.linalg.norm(r_i,2) < tol:
            iter = i
            return [x_i, iter]

        beta = np.dot(r_i,r_i)/np.dot(r_i_1,r_i_1)
        p_i = r_i + beta * p_i_1

        r_i_1 = r_i
        p_i_1 = p_i
        x_i_1 = x_i

    iter = -1
    return [x_i,iter]

if __name__ == "__main__":
    A = np.array([[10, -1, 0], [-1, 10, -2], [0, -2, 10]])
    b = np.array([9, 7, 8]).T
    x0 = 0.1*np.ones(b.shape)
    tol = 1.0e-8
    maxitr = 1000

    [sol, it_num] = cg(A,b,x0,tol,maxitr)
    print('x: \n',sol)
    print(it_num)