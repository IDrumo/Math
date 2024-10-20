import numpy as np

def cholesky_decomposition(A):
    n = A.shape[0]
    S = np.zeros_like(A, dtype=complex)

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                S[i, j] = np.sqrt(A[i, i] - np.sum(S[i, :j] ** 2))
            else:
                S[i, j] = (A[i, j] - np.sum(S[i, :j] * S[j, :j])) / S[j, j]

    return S

def solve_lower_triangular(S, b):
    n = S.shape[0]
    y = np.zeros(n, dtype=complex)

    for i in range(n):
        y[i] = (b[i] - np.sum(S[i, :i] * y[:i])) / S[i, i]

    return y

def solve_upper_triangular(S, y):
    n = S.shape[0]
    x = np.zeros(n, dtype=complex)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.sum(S[i, i + 1:] * x[i + 1:])) / S[i, i]

    return x

def solve_systems(A, b):
    S = cholesky_decomposition(A)
    y = solve_lower_triangular(S, b)
    x = solve_upper_triangular(S.T, y)  # Используем транспонированную матрицу S
    return x, y

if __name__ == '__main__':
    A = np.array([
        [2.2, 4, -3, 1.5, 0.6, 2, 0.7],
        [4, 3.2, 1.5, -0.7, -0.8, 3, 1],
        [-3, 1.5, 1.8, 0.9, 3, 2, 2],
        [1.5, -0.7, 0.9, 2.2, 4, 3, 1],
        [0.6, -0.8, 3, 4, 3.2, 0.6, 0.7],
        [2, 3, 2, 3, 0.6, 2.2, 4],
        [0.7, 1, 2, 1, 0.7, 4, 3.2]
    ], dtype=complex)

    b = np.array([3.2, 4.3, -0.1, 3.5, 5.3, 9.0, 3.7], dtype=complex)

    solution, _ = solve_systems(A, b)

    print("Вектор ответа (собственный метод):", solution)

    norm_custom = np.linalg.norm(A @ solution - b)
    print("Проверка (собственный метод): ||Ax - b|| =", norm_custom)

    solution_builtin = np.linalg.solve(A, b)
    print("Вектор ответа (встроенный метод):", solution_builtin)

    norm_builtin = np.linalg.norm(A @ solution_builtin - b)
    print("Проверка (встроенный метод): ||Ax - b|| =", norm_builtin)

    norm_difference = np.linalg.norm(solution - solution_builtin)
    print("Разница между решениями:", norm_difference)
