import numpy as np
from scipy.optimize import minimize

from subs.lagrange import LagrangeFunction
from subs.vizualizers import SurfaceVisualizer


def f(x, A, b):
    """
    Функция для вычисления значения целевой функции f(x) = 1/2 * (x^T * A * x) - b * x.
    """
    return 0.5 * np.dot(x.T, np.dot(A, x)) - np.dot(b, x)


def is_positive_definite(matrix):
    """
    Проверка, является ли матрица положительно определенной.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False

    for i in range(1, matrix.shape[0] + 1):
        minor = matrix[:i, :i]
        if np.linalg.det(minor) <= 0:
            return False

    return True


def gradient_descent(A, b, x0, alpha=1, eps=1e-10, max_iter=10000):
    """
    Алгоритм градиентного спуска для нахождения минимума функции.

    Args:
        A (np.array): Матрица.
        b (np.array): Вектор.
        x0 (np.array): Начальная точка.
        alpha (float): Шаг.
        eps (float): Точность.
        max_iter (int): Максимальное количество итераций.

    Returns:
        np.array: Минимум.
        float: Значение функции в минимуме.
        int: Количество итераций.
    """
    if not is_positive_definite(A):
        raise ValueError("Матрица A должна быть положительно определенной.")

    x = x0
    iterations = 0

    for i in range(max_iter):
        grad = A @ x - b

        x_new = x - alpha * grad  # Обновляем x

        if f(x_new, A, b) > f(x, A, b):
            alpha /= 2  # Уменьшаем шаг, если значение функции возросло

        if np.linalg.norm(grad) < eps:  # Условие остановки
            break

        x = x_new
        iterations += 1

    return x, f(x, A, b), iterations  # Возвращаем результат и количество итераций


def coordinate_descent(A, b, x0, h=0.1, max_iter=10000, eps=1e-10):
    """
    Алгоритм покоординатного спуска для нахождения минимума функции.

    Args:
        A (np.array): Матрица.
        b (np.array): Вектор.
        x0 (np.array): Начальная точка.
        h (float): Шаг для проверки.
        max_iter (int): Максимальное количество итераций.
        eps (float): Допустимая погрешность.

    Returns:
        np.array: Минимум.
        float: Значение функции в минимуме.
        int: Количество итераций.
    """
    h = [h] * x0.size
    x = x0
    iterations = 0

    for i in range(max_iter):
        old_f = f(x, A, b)

        for j in range(len(x)):
            x_temp = x.copy()

            # Проверяем, увеличится ли функция, если мы изменим j-ю координату
            x_temp[j] += h[j]
            if f(x_temp, A, b) < old_f:
                x = x_temp
                continue

            x_temp[j] -= 2 * h[j]  # Уменьшаем j-ю координату
            if f(x_temp, A, b) < old_f:
                x = x_temp
                continue

            x_temp[j] += h[j]  # Возвращаем j-ю координату обратно

            h[j] /= 2  # Уменьшаем шаг, если не нашли улучшения

        if np.abs(old_f - f(x, A, b)) < eps:  # Условие остановки
            break

        iterations += 1

    return x, f(x, A, b), iterations  # Возвращаем результат и количество итераций


def main():
    # Пример использования
    A = np.array([
        [2, 3, 1],
        [2, 7, 2],
        [1, 3, 3]
    ])
    b = np.array([3, 4, 5])
    x0 = np.array([0 for i in range(b.size)], dtype='float')
    alpha = 0.5
    h = 0.1

    # Выполнение градиентного спуска
    x_min_gd, f_min_gd, iterations_gd = gradient_descent(A, b, x0, alpha)
    print(
        f"Градиентный спуск.\nМинимум достигается в: {x_min_gd}, \nзначение функции: {f_min_gd}, \nколичество итераций: "
        f"{iterations_gd}\n")

    # Выполнение покоординатного спуска
    x_min_cd, f_min_cd, iterations_cd = coordinate_descent(A, b, x0, h)
    print(
        f"Покоординатный спуск.\nМинимум достигается в: {x_min_cd}, \nзначение функции: {f_min_cd}, "
        f"\nколичество итераций: {iterations_cd}\n")

    # Проверка истинного наименьшего значения через встроенный метод минимизации функции
    res = minimize(f, x0, args=(A, b), method='BFGS')
    print(f"Встроенный метод минимизации.\nМинимум достигается в: {res.x}, \nзначение функции: {res.fun}\n")


def main2():
    A = np.array([
        [2, 3, 1],
        [2, 7, 2],
        [1, 3, 3]
    ])
    b = np.array([3, 4, 5])

    # Создаем объект класса LagrangeFunction
    lagrange_func = LagrangeFunction()

    # Устанавливаем целевую функцию
    lagrange_func.set_optimization_function_with_matrix(A, b)

    # Добавляем ограничения массивом
    constraints = [lagrange_func.variables[0] + lagrange_func.variables[1] - 1,
                   lagrange_func.variables[0] - lagrange_func.variables[1]]  # x0 + x1 = 1 и x0 - x1 = 0
    lagrange_func.add_constraints(constraints)


def main3():
    A = np.array([
        [2, 3, 1],
        [2, 7, 2],
        [1, 3, 3]
    ])
    b = np.array([3, 4, 5])

    visualizer = SurfaceVisualizer(A, b)
    visualizer.visualize_surface()


if __name__ == '__main__':
    main()
    main2()
    # main3()
