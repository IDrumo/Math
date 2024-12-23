import numpy as np


def get_inv(A, depth=0):
    n = len(A)  # Размер матрицы
    k = n - 1   # Индекс последней строки/столбца

    # Базовый случай: если матрица 1x1
    if n == 1:
        return np.matrix([[1 / A[0, 0]]])

    # Подматрица, исключающая последнюю строку и столбец
    Ap = A[:k, :k]
    V, U = A[k, :k], A[:k, k].reshape(-1, 1)  # Последняя строка и последний столбец

    # Рекурсивный вызов для нахождения обратной подматрицы
    Ap_inv = get_inv(Ap, depth + 1)

    # Вычисление коэффициентов
    alpha = 1 / (A[k, k] - V * Ap_inv * U).item()  # Определитель
    Q = -V * Ap_inv * alpha  # Вектор Q
    P = Ap_inv - Ap_inv * U * Q  # Обновленная подматрица P
    Z = - Ap_inv * U * alpha  # Вектор Z

    # Создание полной обратной матрицы
    A_inv = np.matrix([[0.0] * n for _ in range(n)])
    A_inv[:k, :k] = P  # Заполнение верхней левой части
    A_inv[k, :k] = Q[0]  # Заполнение последней строки
    A_inv[:k, k] = Z[:, 0]  # Заполнение последнего столбца
    A_inv[k, k] = alpha  # Заполнение нижнего правого элемента

    return A_inv


if __name__ == '__main__':
    A = np.array([
        [0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
        [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
        [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
        [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
        [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
        [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
        [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105],
    ])
    b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])
    x_answer = np.array([11.092, -2.516, 0.721, -2.515, -1.605, 3.624, -4.95])

    # Нахождение обратной матрицы
    A_inv = get_inv(A)
    print("\nОбратная матрица:\n", A_inv)

    # Проверка: произведение A и A_inv должно быть единичной матрицей
    identity_check = np.dot(A, A_inv)
    print("\nПроверка: A * A_inv =\n", identity_check)

    # Вычисление суммарного отклонения от единичной матрицы
    identity_matrix = np.eye(A.shape[0])  # Создание единичной матрицы
    deviation = np.sum(np.abs(identity_check - identity_matrix))  # Суммарное отклонение
    print("\nСуммарное отклонение от единичной матрицы:", deviation)

    # Проверка, что диагональные элементы обратной матрицы равны единице
    diagonal_elements = np.diag(identity_check)
    if np.allclose(diagonal_elements, 1):
        print("Все диагональные элементы равны единице.")
    else:
        print("Некоторые диагональные элементы не равны единице:", diagonal_elements)

    # Решение системы уравнений Ax = b
    x = A_inv * b.reshape(-1, 1)  # Умножение обратной матрицы на вектор b
    print("\nРешение системы Ax = b:\n", x)

    # Проверка невязки
    residual = A @ x - b.reshape(-1, 1)  # Невязка
    print("\nНевязка Ax - b:\n", residual)
    residual_norm = np.linalg.norm(residual)  # Норма невязки
    print("Норма невязки:", residual_norm)
