import numpy as np


def spectral_radius(M: np.ndarray) -> float:
    """
    Compute spectral radius of matrix M.
    """
    eigenvalues = np.linalg.eigvals(M)
    return max(abs(eigenvalues))

