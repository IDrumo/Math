import numpy as np


def max_element_with_coordinates(matrix):
    # Находим индекс максимального по модулю элемента
    index = np.argmax(np.abs(matrix))  # возвращает одномерный индекс

    # Получаем координаты (строка, столбец)
    # через функцию которая по одномерному индексу и форме матрицы
    # возвращает координаты
    coordinates = np.unravel_index(index, matrix.shape)

    # Получаем максимальный по модулю элемент
    max_element = matrix[coordinates]

    return max_element, coordinates


def normalize_line(line, max_elem=0):
    # Проверяем, что максимальный элемент не равен нулю, чтобы избежать деления на нуль
    max_elem = max(np.abs(line)) if max_elem == 0 else max_elem

    # Нормализуем строку, деля каждый элемент на максимальный
    line /= max_elem

    return line


def gauss_step(full_matrix, normalized_line, max_element_coord):
    # Получаем координаты максимального элемента
    matrix, free_column = full_matrix
    max_row, max_col = max_element_coord

    for i in range(matrix.shape[0]):
        if i != max_row:  # Не вычитаем из строки с максимальным элементом
            matrix[i] -= normalized_line * matrix[i, max_col]
            free_column[i] -= free_column[i] * matrix[i, max_col]


def gauss_with_pivoting_first_exemplar(matrix, free_column, x=None):
    """
    Решение системы линейных уравнений Ax = b методом Гаусса с выбором главного элемента.

    :param matrix: Коэффициентная матрица (numpy matrixay)
    :param free_column: Вектор свободных членов (numpy matrixay)
    :return: Решение системы (numpy matrixay)
    """
    if x is None:
        x = []

    if len(free_column) == 0:
        return

    max_elem, coord = max_element_with_coordinates(matrix)
    normalized_line = normalize_line(matrix[coord[0]], max_elem)
    free_column[coord[0]] /= max_elem

    gauss_step((matrix, free_column), normalized_line, coord)

    # Исключаем строку
    matrix_without_row = np.delete(matrix, coord[0], axis=0)
    # Исключаем столбец
    result_matrix = np.delete(matrix_without_row, coord[1], axis=1)
    # Исключаем строку из столбца свободных членов
    result_free_column = np.delete(free_column, coord[0], axis=0)

    gauss_with_pivoting_first_exemplar(result_matrix, result_free_column, x)

    x.append(matrix[coord] - np.sum(matrix[coord[0]] * x) + free_column[coord[0]])


def gauss_with_pivoting(matrix, free_column):

    for k in range(matrix.shape[0] - 1):
        # поиск строки с максимальным элементом
        max_elem = 0
        string_number = 0
        for i in range(k, matrix.shape[0]):
            if abs(matrix[i, k]) > abs(max_elem):
                max_elem = matrix[i, k]
                string_number = i

        # меняем местами строки квадратной матрицы
        matrix[[k, string_number]] = matrix[[string_number, k]]

        # меняем местами элементы вектора-столбца
        free_column[k], free_column[string_number] = free_column[string_number], free_column[k]

        # делим полученную строку на max_elem
        matrix[k] = matrix[k] / max_elem
        free_column[k] = free_column[k] / max_elem

        # домножаем строку на коэффициенты и вычитаем ее из остальных строк
        for i in range(k + 1, matrix.shape[0]):
            multiplier = matrix[i, k]
            matrix[i] = matrix[i] - matrix[k] * multiplier
            free_column[i] = free_column[i] - free_column[k] * multiplier

        # находим аргументы уравнений
    arg = [free_column[free_column.shape[0] - 1] / (matrix[matrix.shape[0] - 1, matrix.shape[0] - 1])]
    for i in range(matrix.shape[0] - 2, -1, -1):
        n = free_column[i]
        for j in range(len(arg)):
            n -= arg[j] * matrix[i, matrix.shape[0] - 1 - j]
        arg.append(n)

    # переворачиваем значения в списке
    return [i for i in reversed(arg)]


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

    print(np.linalg.solve(A1, b1))
    x = gauss_with_pivoting(A, b)
    print(x)

    print(
        "Модуль разности полученного решения и решения через библиотеку np: \n" +
        f"{np.linalg.norm(np.linalg.solve(A1, b1) - x)}"
    )

    # print(A)
    print(
        "Модуль разности произведения матрицы на вектор ответа и вектора свободных членов: \n" +
        f"{np.linalg.norm(A1 @ x - b1)}"
    )
