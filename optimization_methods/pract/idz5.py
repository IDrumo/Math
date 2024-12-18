import numpy as np


def simplex(c, A, b):
    m, n = A.shape
    # Добавляем искусственные переменные
    A = np.hstack((A, np.eye(m)))
    c = np.hstack((c, np.zeros(m)))

    # Начальная базисная матрица
    basis = list(range(n, n + m))

    while True:
        # Определяем коэффициенты в целевой функции
        cb = c[basis]
        y = cb @ np.linalg.inv(A[:, basis])
        reduced_costs = c - y @ A

        # Проверяем, есть ли отрицательные коэффициенты
        if np.all(reduced_costs >= 0):
            # Оптимальное решение найдено
            x = np.zeros(n + m)
            x[basis] = np.linalg.solve(A[:, basis], b)
            return x[:n], cb @ x[basis]  # Возвращаем решение и значение целевой функции

        # Выбираем переменную для ввода в базис
        entering = np.argmin(reduced_costs)

        # Проверяем, как выйти из базиса
        ratios = np.where(A[:, entering] > 0, b / A[:, entering], np.inf)
        leaving = np.argmin(ratios)

        # Обновляем базис
        basis[leaving] = entering

        # Обновляем матрицы
        pivot = A[leaving, entering]
        A[leaving] /= pivot
        b[leaving] /= pivot

        for i in range(m):
            if i != leaving:
                ratio = A[i, entering]
                A[i] -= ratio * A[leaving]
                b[i] -= ratio * b[leaving]


# Пример использования
c = np.array([4, 5])  # Целевая функция
A = np.array(
    [[2, 4],
     [1, 1],
     [2, 1]])  # Ограничения
b = np.array([560, 170, 300])  # Правая часть ограничений

solution, objective_value = simplex(c, A, b)
print("Оптимальное решение:", solution)
print("Значение целевой функции:", objective_value)
