import numpy as np


def decompose_to_LU(A):
    """
    Разложение матрицы A на L и U.

    :param A: Коэффициентная матрица (numpy array)
    :return: LU-матрица (numpy array)
    """
    n = A.shape[0]
    lu_matrix = np.zeros((n, n))

    for k in range(n):
        # Вычисляем элементы строки k
        for j in range(k, n):
            lu_matrix[k, j] = A[k, j] - lu_matrix[k, :k] @ lu_matrix[:k, j]
        # Вычисляем элементы столбца k
        for i in range(k + 1, n):
            lu_matrix[i, k] = (A[i, k] - lu_matrix[i, :k] @ lu_matrix[:k, k]) / lu_matrix[k, k]

    return lu_matrix


def get_L(U):
    """
    Извлечение матрицы L из LU-матрицы.

    :param U: LU-матрица (numpy array)
    :return: Треугольная матрица L (numpy array)
    """
    L = U.copy()
    for i in range(L.shape[0]):
        L[i, i] = 1
        L[i, i + 1:] = 0
    return L


def get_U(U):
    """
    Извлечение матрицы U из LU-матрицы.

    :param U: LU-матрица (numpy array)
    :return: Треугольная матрица U (numpy array)
    """
    U_matrix = U.copy()
    for i in range(1, U_matrix.shape[0]):
        U_matrix[i, :i] = 0
    return U_matrix


def solve_LU(lu_matrix, b):
    """
    Решение системы уравнений с использованием LU-матрицы.

    :param lu_matrix: LU-матрица (numpy array)
    :param b: Вектор свободных членов (numpy array)
    :return: Вектор решений (numpy array)
    """
    n = lu_matrix.shape[0]
    y = np.zeros((n, 1))

    # Решение Ly = b
    for i in range(n):
        y[i] = b[i] - lu_matrix[i, :i] @ y[:i]

    x = np.zeros((n, 1))

    # Решение Ux = y
    for i in range(1, n + 1):
        x[-i] = (y[-i] - lu_matrix[-i, -i:] @ x[-i:, 0]) / lu_matrix[-i, -i]

    return x


if __name__ == '__main__':
    A = np.array([
        [0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
        [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
        [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
        [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
        [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
        [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
        [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]
    ])
    A1 = A.copy()
    b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])
    b1 = b.copy()
    x_answer = np.array([11.092, -2.516, 0.721, -2.515, -1.605, 3.624, -4.95])

    # Разложение на L и U
    LU = decompose_to_LU(A)
    L = get_L(LU)
    U = get_U(LU)

    # Решение системы
    solution = solve_LU(LU, b).flatten()

    print("Решение системы уравнений:", solution)

    print(
        "Модуль разности полученного решения и решения через библиотеку np: \n" +
        f"{np.linalg.norm(np.linalg.solve(A1, b1) - solution)}"
    )

    # print(A)
    print(
        "Модуль разности произведения матрицы на вектор ответа и вектора свободных членов: \n" +
        f"{np.linalg.norm(A1 @ solution - b1)}"
    )
