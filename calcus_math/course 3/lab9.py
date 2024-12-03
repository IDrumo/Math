import numpy as np


# Функция оптимизации
def f_0(x, A, b):
    return 0.5 * np.dot(x.T, np.dot(A, x)) - np.dot(b, x)


# Градиент функции оптимизации
def gradient_f_0(x, A, b):
    return np.dot(A, x) - b


# Метод градиентного спуска
def mu(r, A):
    """
    Функция для вычисления параметра mu, который используется для обновления решения.
    r - остаток (разность между Ax и b)
    A - матрица
    """
    return np.dot(r, A @ r) / np.dot(A @ r, A @ r)


def gradient_descent(A, b, x0, eps=1e-10, max_iterations=1000):
    x = x0

    for i in range(max_iterations):
        r_k = np.dot(A, x) - b  # Остаток
        mu_k = mu(r_k, A)  # Вычисляем mu

        x_new = x - mu_k * gradient_f_0(x, A, b)  # Обновляем x

        # Проверка условия остановки по изменению x
        if np.linalg.norm(x_new - x) < eps:
            return x_new, i  # Возвращаем найденное значение и количество итераций

        x = x_new  # Обновляем x для следующей итерации

    return x, max_iterations  # Если не достигли сходимости, возвращаем последнее значение


# Основной код
if __name__ == "__main__":

    A = np.array([
        [10.9, 1.2, 2.1, 0.9],
        [1.2, 11.2, 1.5, 2.5],
        [2.1, 1.5, 9.8, 1.3],
        [0.9, 2.5, 1.3, 12.1]
    ])

    # Вектор b
    b = np.array([1, 2, 3, 4])

    x0 = np.zeros(b.size)
    x_star, iterations = gradient_descent(A, b, x0, max_iterations=10000)

    print("Решение системы Ax + b = 0 достигается в точке:", x_star)
    print("За ", iterations, " итераций")
    print("Невязка полученного решения:", np.linalg.norm(np.dot(A, x_star) - b))
    print("Истинное посчитанное решение (встроенный метод): ", np.linalg.solve(A, b))
