import numpy as np
from src.jacobi import jacobi

# Example: diagonally dominant system
A = np.array([[4, -1, 0],
              [-1, 4, -1],
              [0, -1, 3]], dtype=float)

b = np.array([15, 10, 10], dtype=float)

x, history = jacobi(A, b)

print("Approximate solution:", x)
print("Final residual:", history[-1])
print("Iterations:", len(history))
