import numpy as np
from scipy.linalg import solve_banded
from prettytable import PrettyTable

# Параметры задачи
a, b = 0, 1  # Интервал [0, 1]
n = 4  # Количество внутренних узлов
h = (b - a) / n  # Шаг сетки

# Создаем расширенную сетку с дополнительными узлами для B-сплайнов
nodes = np.linspace(a - 3 * h, b + 3 * h, n + 7)


# Функции коэффициентов дифференциального уравнения
def p(x): return x + 1
def q(x): return 1
def f(x): return -2 / (x + 1) ** 3 + 1


# Краевые условия
alpha1, beta1, gamma1 = 0, 1, 1  # u'(0) = 1
alpha2, beta2, gamma2 = 1, 0, 0.5  # u(1) = 0.5


# Реализация кубического B-сплайна
def B_spline(x, i, nodes):
    if i < 0 or i >= len(nodes) - 4:
        return 0.0
    x0, x1, x2, x3, x4 = nodes[i], nodes[i + 1], nodes[i + 2], nodes[i + 3], nodes[i + 4]

    if x < x0 or x >= x4:
        return 0.0

    if x < x1:
        t = (x - x0) / (x1 - x0)
        return t ** 3 / 6
    elif x < x2:
        t = (x - x1) / (x2 - x1)
        return (1 + 3 * t + 3 * t ** 2 - 3 * t ** 3) / 6
    elif x < x3:
        t = (x - x2) / (x3 - x2)
        return (4 - 6 * t ** 2 + 3 * t ** 3) / 6
    else:
        t = (x - x3) / (x4 - x3)
        return (1 - t) ** 3 / 6


# Первая производная B-сплайна
def B_prime(x, i, nodes):
    if i < 0 or i >= len(nodes) - 4:
        return 0.0
    x0, x1, x2, x3, x4 = nodes[i], nodes[i + 1], nodes[i + 2], nodes[i + 3], nodes[i + 4]

    if x < x0 or x >= x4:
        return 0.0

    if x < x1:
        t = (x - x0) / (x1 - x0)
        return (3 * t ** 2) / (6 * (x1 - x0))
    elif x < x2:
        t = (x - x1) / (x2 - x1)
        return (3 + 6 * t - 9 * t ** 2) / (6 * (x2 - x1))
    elif x < x3:
        t = (x - x2) / (x3 - x2)
        return (-12 * t + 9 * t ** 2) / (6 * (x3 - x2))
    else:
        t = (x - x3) / (x4 - x3)
        return (-3 * (1 - t) ** 2) / (6 * (x4 - x3))


# Вторая производная B-сплайна
def B_double_prime(x, i, nodes):
    if i < 0 or i >= len(nodes) - 4:
        return 0.0
    x0, x1, x2, x3, x4 = nodes[i], nodes[i + 1], nodes[i + 2], nodes[i + 3], nodes[i + 4]

    if x < x0 or x >= x4:
        return 0.0

    if x < x1:
        t = (x - x0) / (x1 - x0)
        return (6 * t) / (6 * (x1 - x0) ** 2)
    elif x < x2:
        t = (x - x1) / (x2 - x1)
        return (6 - 18 * t) / (6 * (x2 - x1) ** 2)
    elif x < x3:
        t = (x - x2) / (x3 - x2)
        return (-12 + 18 * t) / (6 * (x3 - x2) ** 2)
    else:
        t = (x - x3) / (x4 - x3)
        return (6 * (1 - t)) / (6 * (x4 - x3) ** 2)


# Количество коэффициентов (n+3 базисных функций)
n_coeff = n + 3
A = np.zeros((n_coeff, n_coeff))
F = np.zeros(n_coeff)

# Заполнение матрицы для внутренних точек коллокации
for k in range(n + 1):
    xk = a + k * h
    for i in range(-1, n + 1):
        col = i + 1
        # Оператор L = u'' + (x+1)u' + u
        A[k, col] = B_double_prime(xk, i, nodes) + (xk + 1) * B_prime(xk, i, nodes) + B_spline(xk, i, nodes)
    F[k] = f(xk)

# Граничное условие u'(0) = 1
row = n + 1
for i in range(-1, n + 1):
    A[row, i + 1] = B_prime(0, i, nodes)
F[row] = 1

# Граничное условие u(1) = 0.5
row += 1
for i in range(-1, n + 1):
    A[row, i + 1] = B_spline(1, i, nodes)
F[row] = 0.5

# Решение системы уравнений
b_coeff = np.linalg.lstsq(A, F, rcond=None)[0]


# Приближенное решение
def S(x, nodes, b_coeff):
    result = 0.0
    for i in range(-1, n + 1):
        result += b_coeff[i + 1] * B_spline(x, i, nodes)
    return result


# Точное решение
def u_exact(x):
    return x / (x + 1)


# Создание таблицы для сравнения результатов
table = PrettyTable()
table.field_names = ["x", "Приближенное", "Точное", "Ошибка"]

x_vals = np.linspace(0, 1, 6)
for x in x_vals:
    approx = S(x, nodes, b_coeff)
    exact = u_exact(x)
    table.add_row([f"{x:.2f}", f"{approx:.6f}", f"{exact:.6f}", f"{abs(approx - exact):.2e}"])

print("Результаты решения методом коллокации с B-сплайнами:")
print(table)

# Проверка краевых условий
print("\nПроверка краевых условий:")
print(f"u'(0) = {sum(b_coeff[i + 1] * B_prime(0, i, nodes) for i in range(-1, n + 1)):.6f} (ожидается 1.0)")
print(f"u(1) = {S(1, nodes, b_coeff):.6f} (ожидается 0.5)")