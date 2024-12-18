import numpy as np


# Обратная итерация для поиска собственных значений
def inverse_iteration2(A, x0, epsilon=1e-10, max_iterations=1000):
    n = A.shape[0]
    x = x0 / np.linalg.norm(x0)  # Нормализация начального вектора
    lambda_prev = None

    for k in range(max_iterations):
        # Решение системы уравнений (A - lambda_k * I)x = 0
        # Здесь мы используем x в качестве начального вектора
        b = x
        x = np.linalg.solve(A, b)

        # Нормализация вектора
        x_norm = np.linalg.norm(x)
        if x_norm == 0:
            raise ValueError("Нормированный вектор равен нулю. Проверьте матрицу A.")
        x = x / x_norm  # Нормализация вектора

        # Вычисление нового собственного значения
        lambda_k = np.dot(x, np.dot(A, x))  # Используем формулу для вычисления собственного значения

        # Проверка условия остановки
        if lambda_prev is not None and np.abs(lambda_k - lambda_prev) < epsilon:
            break

        lambda_prev = lambda_k

    return lambda_k, x, k


def inverse_iteration(A, x0, epsilon=1e-10, max_iterations=1000):
    """
        Метод прямых итераций для нахождения наименьшего собственного значения и соответствующего собственного вектора матрицы.

        Args:
          A: Матрица.
          x0: Начальный вектор.
          eps: Точность вычислений.
          max_iter: Максимальное количество итераций.

        Returns:
          Кортеж из двух элементов: наименьшее собственное значение и соответствующий собственный вектор.
        """
    x = x0  # Начальный вектор
    alpha_prev = np.max(np.abs(x))

    for k in range(max_iterations):
        # Решение системы уравнений Ax = x / alpha_prev
        # В методичке описано, что можно использовать треугольную матрицу для вычисления, чтобы оптимизировать код,
        # но в данной работе я решил не акцентировать на этом внимание и просто использую встроенный метод
        # под капотом у метода совокупность методов, выбор которых для решения зависит от вида матрицы
        # в общем случае это метод гаусса
        b = x / alpha_prev
        x = gauss_elimination(A, b)

        # Определение alpha как наибольшей компоненты вектора
        alpha = np.max(np.abs(x))

        # Проверка условия остановки
        if np.abs(alpha - alpha_prev) < epsilon:
            break

        alpha_prev = alpha

    # Вычисление собственного значения
    lambda_k = 1 / alpha
    return lambda_k, x, k


def gauss_elimination(A, b):
    """Решение системы линейных уравнений Ax = b методом Гаусса."""
    n = len(b)

    # Объединяем матрицу A и вектор b в расширенную матрицу
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))

    # Прямой ход метода Гаусса
    for i in range(n):
        # Поиск максимального элемента в текущем столбце
        max_row_index = np.argmax(np.abs(augmented_matrix[i:n, i])) + i
        augmented_matrix[[i, max_row_index]] = augmented_matrix[[max_row_index, i]]  # Поменять строки местами

        # Обнуление элементов под текущим pivot
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:n])) / augmented_matrix[i, i]

    return x


def main():
    matrices = {
        "Матрица 1": (np.array([[-0.168700, 0.353699, 0.008540, 0.733624],
                                [0.353699, 0.056519, -0.723182, -0.076440],
                                [0.008540, -0.723182, 0.015938, 0.342333],
                                [0.733624, -0.076440, 0.342333, -0.045744]]),
                      np.array([-0.943568, -0.744036, 0.687843, 0.857774])),
        "Матрица 2": (np.array([[2.2, 1, 0.5, 2],
                                [1, 1.3, 2, 1],
                                [0.5, 2, 0.5, 1.6],
                                [2, 1, 1.6, 2]]),
                      np.array([5.652, 1.545, -1.420, 0.2226])),
        "Матрица 3": (np.array([[1.00, 0.42, 0.54, 0.66],
                                [0.42, 1.00, 0.32, 0.44],
                                [0.54, 0.32, 1.00, 0.22],
                                [0.66, 0.44, 0.22, 1.00]]),
                      np.array([2.3227, 0.7967, 0.6383, 0.2423])),
        "Проверяющая матрица": (np.array(
            [[2.00, 1.00],
             [1.00, 2.00]]),
                                np.array([4, 5])),
        "Матрица методички": (np.array(
            [[2, 1, 1],
             [1, 2.5, 1],
             [1, 1, 3]
             ]),

                              np.array([1.185089, 4.555030, 1.759839])),
    }

    for name, (A, expected_eigenvalues) in matrices.items():
        print(f"-------------{name}-------------")

        x0 = np.array([1 for i in range(expected_eigenvalues.size)])

        eigenval, vector, iterations = inverse_iteration(A, x0)

        residual_matrix = A - eigenval * np.eye(A.shape[0])
        det = np.linalg.det(residual_matrix)

        # Выводим результаты
        print("Истинные собственные значения:", np.sort(np.linalg.eigvals(A)))
        print("Найденное собственное значение: ", eigenval)
        print("Найденный собственный вектор: ", vector)
        print("Количество итераций: ", iterations)
        print("Определитель матрицы |A - λI| (должен быть близок к нулю):", det)
        print()


if __name__ == '__main__':
    main()
