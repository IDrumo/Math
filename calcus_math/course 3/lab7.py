import numpy as np


def qr(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        P = np.eye(m)
        P[i:, i:] = make_piece_P(A[i:, i])
        Q = np.dot(Q, P)
        A = np.dot(P, A)
    return Q, A


def make_piece_P(a):
    p = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    p[0] = 1
    P = np.eye(a.shape[0])
    P -= (2 / np.dot(p, p)) * np.dot(p[:, None], p[None, :])
    return P


def get_answer(A, b):
    Q, R = qr(A)

    # Вычисляем Q^T * b
    b_tilde = np.dot(Q.T, b)

    # Инициализируем вектор x для хранения решения
    n = R.shape[1]
    x = np.zeros(n)

    # Метод обратной подстановки для решения Rx = Q^T b
    for i in range(n - 1, -1, -1):
        x[i] = (b_tilde[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]

    return x


if __name__ == '__main__':
    A = np.array([
        [2.2, 4, -3, 1.5, 0.6, 2, 0.7],
        [4, 3.2, 1.5, -0.7, -0.8, 3, 1],
        [-3, 1.5, 1.8, 0.9, 3, 2, 2],
        [1.5, -0.7, 0.9, 2.2, 4, 3, 1],
        [0.6, -0.8, 3, 4, 3.2, 0.6, 0.7],
        [2, 3, 2, 3, 0.6, 2.2, 4],
        [0.7, 1, 2, 1, 0.7, 4, 3.2]
    ])

    # Вектор b - вектор свободных членов системы уравнений
    b = np.array([3.2, 4.3, -0.1, 3.5, 5.3, 9.0, 3.7])
    x = get_answer(A, b)

    print("\nВектор ответа (собственный метод):", x)

    norm_custom = np.linalg.norm(A @ x - b)
    print("\nПроверка (собственный метод): ||Ax - b|| =", norm_custom)

    # Проверка невязки
    residual = A @ x - b  # Невязка
    print("\nНевязка (Ax - b):", residual)

