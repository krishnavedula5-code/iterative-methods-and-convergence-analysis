import numpy as np
import matplotlib.pyplot as plt
from src.jacobi import jacobi
from experiments.spectral_radius_analysis import spectral_radius


# Example 1: Strictly Diagonally Dominant (Should Converge)
A1 = np.array([[4, -1, 0],
               [-1, 4, -1],
               [0, -1, 3]], dtype=float)

b1 = np.array([15, 10, 10], dtype=float)

x1, history1 = jacobi(A1, b1)

D1 = np.diag(np.diag(A1))
R1 = A1 - D1
rho1 = spectral_radius(np.linalg.inv(D1) @ R1)

print("Example 1: Diagonally Dominant")
print("Spectral radius:", rho1)
print("Final residual:", history1[-1])
print("Iterations:", len(history1))


# Example 2: Not Diagonally Dominant (May Diverge)
A2 = np.array([[1, 2],
               [3, 4]], dtype=float)

b2 = np.array([1, 1], dtype=float)

x2, history2 = jacobi(A2, b2)

D2 = np.diag(np.diag(A2))
R2 = A2 - D2
rho2 = spectral_radius(np.linalg.inv(D2) @ R2)

print("\nExample 2: Not Diagonally Dominant")
print("Spectral radius:", rho2)
print("Final residual:", history2[-1])
print("Iterations:", len(history2))


# Plot convergence
plt.semilogy(history1, label="Diagonally Dominant")
plt.semilogy(history2, label="Not Diagonally Dominant")
plt.xlabel("Iteration")
plt.ylabel("Residual Norm (log scale)")
plt.title("Jacobi Convergence Behavior")
plt.legend()
plt.grid(True)
plt.show()
