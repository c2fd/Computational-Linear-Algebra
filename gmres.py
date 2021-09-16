import numpy as np

def arnoldi(A, x0, m):
    n = A.shape[0]
    q = x0/np.linalg.norm(x0,2)

    H = np.zeros((m+1,m))
    Q = np.zeros((n,m+1))
    Q[:, 0:1] = q[:,0:1]

    for it1 in np.arange(m):
        w = A@q
        for it2 in np.arange(it1+1):
            H[it2,it1] = np.dot(Q[:,it2],w)
            w = w - H[it2,it1] * Q[:,it2]

        w_norm = np.linalg.norm(w,2)
        H[it1 + 1, it1] = w_norm
        if np.abs(w_norm) < 1.0e-10:
            print('Unfinished!')
            return [Q, H]
        q = w/w_norm
        Q[:, it1 + 1:it1+2] = q[:,0:1]
    return [Q, H]



def gmres(A,b,x0_,m,tol,maxiter):

    x0 = np.array(x0_)
    print('x0',x0.shape)
    iter_gmres = 1

    while iter_gmres < maxiter:
        r = b - A@x0
        [Q, H] = arnoldi(A,r,m)

        beta = np.linalg.norm(r,2)
        e1 = np.zeros((m+1,1))
        e1[0] = 1
        # ym, residuals, rank, sv = np.linalg.lstsq(H,beta*e1,rcond=None)
        ym = np.linalg.lstsq(H,beta*e1,rcond=None)[0]
        xm = x0 + Q[:,:-1]@ym
        r = np.linalg.norm(b-A@xm, 2)
        if r < tol:
            return [xm, r, iter_gmres]
        x0 = xm
        iter_gmres = iter_gmres + 1

    iter_gmres = -1
    return [xm, r, iter_gmres]


if __name__ == "__main__":
    A = np.array([[-4, 1, 0, 0, 1], [1, -4, 1, 0, 0], [0, 1, -4, 1, 0], [0, 0, 1, -4, 1], [1, 0, 0, 1, -4]])
    b = np.ones((A.shape[0],1)) * 2
    x0 = np.ones(b.shape)
    tol = 1.0e-8
    m = 2
    maxit = 1000

    [sol, residual, it_num] = gmres(A, b, x0, m, tol,maxit)
    print('x: \n', sol)
    print(it_num)