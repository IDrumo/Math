import numpy as np


# Обратная итерация для поиска собственных значений
def inverse_iteration2(A, x0, epsilon=1e-6, max_iterations=1000):
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


def inverse_iteration(A, x0, epsilon=1e-6, max_iterations=1000):
    x = x0  # Начальный вектор
    alpha_prev = np.max(np.abs(x))

    for k in range(max_iterations):
        # Решение системы уравнений Ax = x / alpha_prev
        # В методичке описано, что можно использовать треугольную матрицу для вычисления, чтобы оптимизировать код,
        # но в данной работе я решил не акцентировать на этом внимание и просто использую встроенный метод
        # под капотом у метода совокупность методов, выбор которых для решения зависит от вида матрицы
        # в общем случае это метод гаусса
        b = x / alpha_prev
        x = np.linalg.solve(A, b)

        # Определение alpha как наибольшей компоненты вектора
        alpha = np.max(np.abs(x))

        # Проверка условия остановки
        if np.abs(alpha - alpha_prev) < epsilon:
            break

        alpha_prev = alpha

    # Вычисление собственного значения
    lambda_k = 1 / alpha
    return lambda_k, x, k


def main():
    A = np.array([[1.00, 0.42, 0.54, 0.66],
                  [0.42, 1.00, 0.32, 0.44],
                  [0.54, 0.32, 1.00, 0.22],
                  [0.66, 0.44, 0.22, 1.00]])
    b = np.array([1, 1, 1, 1])
    x0 = np.array([1, 1, 1, 1])

    eigenval, vector, iterations = inverse_iteration(A, x0)

    residual_matrix = A - eigenval * np.eye(A.shape[0])
    det = np.linalg.det(residual_matrix)

    # Выводим результаты
    print("Найденное собственное значение: ", eigenval)
    print("Найденный собственный вектор: ", vector)
    print("Количество итераций: ", iterations)
    print("Определитель матрицы |A - λI| (должен быть близок к нулю):", det)
    print("Истинные собственные значения: ", np.linalg.eigvals(A))


if __name__ == '__main__':
    main()
