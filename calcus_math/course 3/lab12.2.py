import numpy as np


# Прямая итерация для поиска собственных значений
def direct_iterations(A, x0, eps=1e-10, max_iter=1000):
    """
    Метод прямых итераций для нахождения наибольшего собственного значения и соответствующего собственного вектора матрицы.

    Args:
      A: Матрица.
      x0: Начальный вектор.
      eps: Точность вычислений.
      max_iter: Максимальное количество итераций.

    Returns:
      Кортеж из двух элементов: наибольшее собственное значение и соответствующий собственный вектор.
    """

    global steps, alpha
    n = A.shape[0]
    x = x0
    for i in range(max_iter):
        steps = i
        x_prev = x
        x = A @ x
        alpha = np.max(np.abs(x))
        x = x / alpha
        if np.linalg.norm(x - x_prev) < eps:
            break

    return alpha, x, steps


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

        eigenval, vector, iterations = direct_iterations(A, x0)

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
