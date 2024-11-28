import numpy as np


# Простая итерация для поиска собственных значений
def power_iteration(A, x0, eps=1e-6, max_iter=100):
    """
    Реализует метод простой итерации (метод степеней) для нахождения
    наибольшего по модулю собственного значения и соответствующего
    собственного вектора матрицы A.

    Args:
      A: Матрица, для которой нужно найти собственное значение.
      x0: Начальное приближение собственного вектора.
      eps: Допустимая погрешность для остановки итераций.
      max_iter: Максимальное число итераций.

    Returns:
      Кортеж из двух элементов: (собственное значение, собственный вектор).
    """

    global l, step
    x = x0
    for i in range(max_iter):
        step = i
        # y = Ax
        y = np.dot(A, x)
        # lambda = y.T @ x
        l = np.dot(y.transpose(), x)
        # x = y / ||y||
        x = y / np.linalg.norm(y)

        # Проверяем критерий остановки
        if np.linalg.norm(x - x0) < eps:
            return l, x, step
        x0 = x

    print("Превышено максимальное число итераций.")
    return l, x, step


def main():
    A = np.array([[1.00, 0.42, 0.54, 0.66],
                  [0.42, 1.00, 0.32, 0.44],
                  [0.54, 0.32, 1.00, 0.22],
                  [0.66, 0.44, 0.22, 1.00]])
    b = np.array([1, 1, 1, 1])
    x0 = [1, 1, 1, 1]

    eigenval, vector, iterations = power_iteration(A, x0)

    residual_matrix = A - eigenval * np.eye(A.shape[0])
    det = np.linalg.det(residual_matrix)

    # Выводим результаты
    print("Найденное собственное значение: ", eigenval)
    print("Найденный собственный вектор: ", vector)
    print("Количество итераций: ", iterations)
    print("Определитель матрицы |A - λI| (должен быть близок к нулю):", det)


if __name__ == '__main__':
    main()
