import numpy as np

def jacobi(A, b, x0=None, tol=1e-8, max_iter=1000):
    """
    Solve Ax = b using the Jacobi iterative method.

    Parameters:
        A : numpy.ndarray
            Coefficient matrix (n x n)
        b : numpy.ndarray
            Right-hand side vector (n,)
        x0 : numpy.ndarray
            Initial guess (n,)
        tol : float
            Convergence tolerance (||x_k+1 - x_k||)
        max_iter : int
            Maximum number of iterations

    Returns:
        x : numpy.ndarray
            Approximate solution
        history : list
            Residual norms at each iteration
    """

    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    D = np.diag(np.diag(A))
    R = A - D

    D_inv = np.linalg.inv(D)

    history = []

    for k in range(max_iter):
        x_new = D_inv @ (b - R @ x)

        residual = np.linalg.norm(A @ x_new - b)
        history.append(residual)

        if np.linalg.norm(x_new - x) < tol:
            return x_new, history

        x = x_new

    return x, history

