import numpy as np

def Gauss_Seidel(A,b):
    m, n = A.shape
    assert(m == n)
    assert(np.diag(A).any())

    x = np.zeros([n,1])
    max_loop = 100
    min_tol = 1.0e-12

    for k in np.arange(max_loop):
        x_old = np.array(x)

        for i in np.arange(n):
            x[i] = 1. / A[i, i] * b[i]
            for j in np.arange(i):
                x[i] += - 1./A[i,i] * A[i,j] * x[j]
            for j in np.arange(i+1,n):
                x[i] += - 1. / A[i,i] * A[i,j] * x_old[j]

        tol = np.linalg.norm(x-x_old,2)
        print('round: ', k, 'lol: ', tol)

        if tol < min_tol:
            print('converge!')
            break
    return x


def Gauss_Seidel_M(A, b):
    m, n = A.shape
    assert(m == n)
    assert(np.diag(A).any())

    D  = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    L = - np.matmul(D_inv, np.tril(A,-1))
    U = - np.matmul(D_inv, np.triu(A, 1))

    G = np.matmul( np.linalg.inv( np.eye(n) - L) , U)
    g = np.matmul( np.linalg.inv( np.eye(n) - L), np.matmul(D_inv,b) )

    #G = np.identity(n) - np.matmul(D_inv, A)
    #g = np.matmul(D_inv, b)

    x = np.zeros(n)
    max_loop = 100
    min_tol = 1.0e-12

    for k in np.arange(max_loop):
        x_old = np.array(x)
        x = np.matmul(G, x_old) + g

        tol = np.linalg.norm(x-x_old,2)
        print('round: ', k, 'lol: ', tol)

        if tol < min_tol:
            print('converge!')
            break
    return x


if __name__== "__main__":
    #A = np.array([[2,1],[1,2]])
    #b = np.array([1,2]).T

    A = np.array([[10,-1,0],[-1,10,-2],[0,-4,10]])
    b = np.array([9,7,6]).T

    print(b.shape)


    y = Gauss_Seidel(A,b)
    #y = Gauss_Seidel_M(A,b)
    print("Gauss_Seidel \n",y)

    z = np.linalg.solve(A,b)
    print('np.linalg \n',z)