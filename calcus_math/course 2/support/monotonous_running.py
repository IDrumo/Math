import numpy as np


def tridiagonal_solver(A, b):
    n = len(A)
    alpha = np.zeros(n)
    beta = np.zeros(n)

    alpha[0] = -A[0, 1] / A[0, 0]
    beta[0] = b[0] / A[0, 0]

    for i in range(1, n - 1):
        denominator = A[i, i] + A[i, i - 1] * alpha[i - 1]
        alpha[i] = -A[i, i + 1] / denominator
        beta[i] = (b[i] - A[i, i - 1] * beta[i - 1]) / denominator

    # Последнее уравнение обрабатываем отдельно
    x = np.zeros(n)
    x[n - 1] = (b[n - 1] - A[n - 1, n - 2] * beta[n - 2]) / (A[n - 1, n - 1] + A[n - 1, n - 2] * alpha[n - 2])
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x


def checker(matrix, answer, resolve):
    result = [False] * len(matrix)
    for i in range(len(matrix)):
        sum = 0
        for j in range(len(matrix[i])):
            sum += matrix[i, j] * resolve[j]
        result[i] = abs(sum - answer[i]) < 1e-6

    return result
