import numpy as np


# Прямая итерация для поиска собственных значений
def direct_iterations(A, x0, eps=1e-6, max_iter=100):
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
    A = np.array([[1.00, 0.42, 0.54, 0.66],
                  [0.42, 1.00, 0.32, 0.44],
                  [0.54, 0.32, 1.00, 0.22],
                  [0.66, 0.44, 0.22, 1.00]])
    b = np.array([1, 1, 1, 1])
    x0 = [1, 1, 1, 1]

    eigenval, vector, iterations = direct_iterations(A, x0)

    residual_matrix = A - eigenval * np.eye(A.shape[0])
    det = np.linalg.det(residual_matrix)

    # Выводим результаты
    print("Найденное собственное значение: ", eigenval)
    print("Найденный собственный вектор: ", vector)
    print("Количество итераций: ", iterations)
    print("Определитель матрицы |A - λI| (должен быть близок к нулю):", det)


if __name__ == '__main__':
    main()
