import numpy as np


def generate_hurwitz_matrix(coefficients):
    """
    Формирует матрицу Гурвица на основе переданного списка коэффициентов.

    :param coefficients: Список коэффициентов многочлена.
    :return: Матрица Гурвица.
    """
    n = len(coefficients) - 1
    hurwitz_matrix = np.zeros((n, n))

    # Заполнение матрицы Гурвица
    for i in range(n):
        for j in range(n):
            if j == 0:
                hurwitz_matrix[i, j] = coefficients[n - i]
            elif i + j == n - 1:
                hurwitz_matrix[i, j] = coefficients[0]
            elif i + j < n - 1:
                hurwitz_matrix[i, j] = 0
            else:
                hurwitz_matrix[i, j] = coefficients[i + j - n + 1]

    return hurwitz_matrix


def main_minors(matrix):
    """
    Находит главные миноры переданной матрицы.

    :param matrix: Входная матрица.
    :return: Список главных миноров.
    """
    n = matrix.shape[0]
    minors = []

    for i in range(1, n + 1):
        minor = np.linalg.det(matrix[:i, :i])
        minors.append(minor)

    return minors


# Пример использования
coefficients = [1, 2, 4, 3, 2]
# coefficients = [2, 3, 4, 2, 1]
hurwitz_matrix = generate_hurwitz_matrix(coefficients)
print("Матрица Гурвица:")
print(hurwitz_matrix)

minors = main_minors(hurwitz_matrix)
print("Главные миноры:")
print(minors)