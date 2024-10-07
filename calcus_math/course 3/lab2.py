import numpy as np


def optimal_gauss_elimination(A, b):
    """
    Решение системы линейных уравнений Ax = b методом оптимального исключения Гаусса.

    :param A: Коэффициентная матрица (numpy array)
    :param b: Вектор свободных членов (numpy array)
    :return: Решение системы (numpy array)
    """
    n = len(b)
    # Создаем расширенную матрицу [A|b]
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])

    for k in range(n):
        # Обнуляем элементы в k-ой строке слева от диагонали
        for i in range(k):
            augmented_matrix[k] -= augmented_matrix[i] * augmented_matrix[k][i]

        # Получаем ведущий коэффициент
        leading_coefficient = augmented_matrix[k][k]

        # Проверяем на нулевой ведущий коэффициент
        if leading_coefficient == 0:
            return -1

        # Нормализуем k-ю строку
        augmented_matrix[k] /= leading_coefficient

        # Обнуляем элементы над диагональю в k-ой строке
        for i in range(k):
            augmented_matrix[i] -= augmented_matrix[k] * augmented_matrix[i][k]

        # print(augmented_matrix)

    # Извлекаем решение из последнего столбца расширенной матрицы
    return augmented_matrix[:, -1]


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

    solution = optimal_gauss_elimination(A, b)

    print("Решение системы уравнений:", solution)
