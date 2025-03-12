import numpy as np


def simple_iteration(A, b, x0=None, eps=1e-6, max_iter=1000, tau=None):
    """
    Решение СЛАУ методом простой итерации.

    Параметры:
    A (np.ndarray): Матрица коэффициентов системы.
    b (np.ndarray): Вектор свободных членов.
    x0 (np.ndarray): Начальное приближение решения.
    eps (float): Желаемая точность решения.
    max_iter (int): Максимальное число итераций.

    Возвращает:
    np.ndarray: Решение СЛАУ.
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)

    x = np.copy(x0)

    if tau is None:
        np.linalg.eigvals(A)
        tau = 2 / (min + max)

    for iteration in range(max_iter):
        # Вычисление нового приближения
        x_new = (tau * b - tau * np.dot(A, x) + np.diag(A) * x) / np.diag(A)

        if np.linalg.norm(x_new - x) < eps:
            return x_new, iteration

        x = x_new

    print(f"Метод простой итерации не сошелся за {max_iter} итераций.")
    return x, max_iter


def relaxation_method2(A, b, omega, x0=None, tol=1e-6, max_iter=1000):
    """
    Решение СЛАУ методом релаксации.

    Параметры:
    A (np.ndarray): Матрица коэффициентов системы.
    b (np.ndarray): Вектор свободных членов.
    x0 (np.ndarray): Начальное приближение решения.
    omega (float): Параметр релаксации (0 < omega < 2).
    tol (float): Желаемая точность решения.
    max_iter (int): Максимальное число итераций.

    Возвращает:
    tuple: Решение СЛАУ и количество итераций.
    """
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n)

    x = np.copy(x0)

    for iteration in range(max_iter):
        # Вычисление нового приближения с учетом релаксации
        x_new = (1 - omega) * x + omega * (b - np.dot(A, x) + np.diag(A) * x) / np.diag(A)

        if np.linalg.norm(x_new - x) < tol:
            return x_new, iteration + 1  # Возвращаем решение и количество итераций
        x = x_new

    print(f"Метод релаксации не сошелся за {max_iter} итераций.")
    return x, max_iter  # Возвращаем последнее приближение и максимальное количество итераций


def relaxation_method(A, b, omega, x0=None, eps=1e-6, max_iter=1000):
    """
    Решает систему линейных уравнений Ax = b методом релаксации.

    :param A: Коэффициентная матрица (numpy.ndarray)
    :param b: Вектор правой части (numpy.ndarray)
    :param omega: Параметр релаксации (float)
    :param x0: Начальное приближение (numpy.ndarray), если None, то будет использован нулевой вектор
    :param eps: Допуск для остановки (float)
    :param max_iter: Максимальное количество итераций (int)
    :return: Приближенное решение (numpy.ndarray)
    """

    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n)

    x = x0.copy()

    for k in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sum1 - sum2)

        # Проверка на сходимость
        if np.linalg.norm(x_new - x) < eps:
            return x_new, k

        x = x_new

    print(f"Метод релаксации не сошелся за {max_iter} итераций.")
    return x, max_iter


if __name__ == '__main__':
    A = np.array([
        [4, -1, 0, 0],
        [-1, 4, -1, 0],
        [0, -1, 4, -1],
        [0, 0, -1, 3]
    ])

    b = np.array([15, 10, 10, 10])

    # A = np.array([
    #     [10, 2, 1],
    #     [1, 10, 2],
    #     [1, 1, 10]
    # ])
    #
    # b = np.array([10, 12, 8])

    omegas = [0.01, 0.5, 1, 1.5, 1.99]

    # Решение методом релаксации для различных значений omega
    for omega in omegas:
        x, iterations = relaxation_method(A, b, omega)
        print(f"omega = {omega}: Решение = {x}, Количество итераций = {iterations}\n")

    # Решение методом простой итерации
    x, iterations = simple_iteration(A, b)
    print(f"Решение методом простой итерации: {x} Количество итераций: {iterations}\n")

    # Решение СЛАУ с использованием встроенного метода NumPy
    x_numpy = np.linalg.solve(A, b)
    print("Решение с использованием np.linalg.solve:", x_numpy)
