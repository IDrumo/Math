import numpy as np
from scipy.linalg import solve_banded
from prettytable import PrettyTable

# Параметры задачи
a, b = 0, 1  # Интервал [0, 1]
n = 4  # Количество внутренних узлов
h = (b - a) / n  # Шаг сетки

# Создаем расширенную сетку с дополнительными узлами
nodes = np.linspace(a - 3 * h, b + 3 * h, n + 7)


# Функции коэффициентов ДУ
def p(x): return x + 1
def q(x): return 1
def f(x): return -2 / (x + 1) ** 3 + 1


# Краевые условия
alpha1, beta1, gamma1 = 0, 1, 1  # u'(0) = 1
alpha2, beta2, gamma2 = 1, 0, 0.5  # u(1) = 0.5

# Вычисляем коэффициенты системы для внутренних узлов
# A = np.zeros(n + 3)
# C = np.zeros(n + 3)
# D = np.zeros(n + 3)
# F = np.zeros(n + 3)
A = np.zeros(n + 1)
C = np.zeros(n + 1)
D = np.zeros(n + 1)
F = np.zeros(n + 1)

for k in range(0, n + 1):
    xk = a + k * h
    A[k] = (1 - 0.5 * p(xk) * h + q(xk) * h ** 2 / 6) / (3 * h)
    D[k] = (1 + 0.5 * p(xk) * h + q(xk) * h ** 2 / 6) / (3 * h)
    C[k] = -A[k] - D[k] + q(xk) * (2 * h) / 6
    F[k] = f(xk) * (2 * h) / 6

# Граничные условия (исправленные)
A[-1], C[-1], D[-1], F[-1] = alpha2 * h - 3 * beta2, 2 * alpha2 * (h + h), alpha2 * h + 3 * beta2, 2 * gamma2 * (
            2 * h + h)
A[-2], C[-2], D[-2], F[-2] = alpha1 * h - 3 * beta1, 2 * alpha1 * (h + h), alpha1 * h + 3 * beta1, 2 * gamma1 * (
            2 * h + h)

# Строим трехдиагональную матрицу (исправленный размер)
matrix = np.zeros((3, n + 1))  # Размер соответствует количеству уравнений

# Заполняем матрицу (исправленный порядок)
matrix[0, 1:] = D[:-1]  # Верхняя диагональ
# matrix[1, :] = C[:n + 1]  # Главная диагональ
matrix[1, :] = C  # Главная диагональ
# matrix[2, :-1] = A[1:n + 1]  # Нижняя диагональ
matrix[2, :-1] = A[1:]  # Нижняя диагональ

# Учет граничных условий (упрощенный вариант)
matrix[1, 0] = 1  # Для u'(0) = 1
matrix[1, -1] = 1  # Для u(1) = 0.5
F[0] = 1
F[-1] = 0.5

# Решаем систему методом прогонки
try:
    # b_coeff = solve_banded((1, 1), matrix, F[:n + 1])
    b_coeff = solve_banded((1, 1), matrix, F)
except np.linalg.LinAlgError:
    # Альтернативный метод если матрица вырождена
    b_coeff = np.linalg.lstsq(matrix.T, F[:n + 1], rcond=None)[0]


# Функция для вычисления кубического B-сплайна
def B_spline(x, i, nodes):
    if i < 0 or i >= len(nodes) - 4:
        return 0.0
    if x < nodes[i] or x >= nodes[i + 4]:
        return 0.0

    t = (x - nodes[i]) / (nodes[i + 1] - nodes[i])

    if nodes[i] <= x < nodes[i + 1]:
        return t ** 3 / 6
    elif nodes[i + 1] <= x < nodes[i + 2]:
        return (1 + 3 * t * (1 - t)) * t ** 2 / 6
    elif nodes[i + 2] <= x < nodes[i + 3]:
        return (1 + 3 * (1 - t) * (1 - (1 - t))) * (1 - t) ** 2 / 6
    elif nodes[i + 3] <= x < nodes[i + 4]:
        return (1 - t) ** 3 / 6
    return 0.0


# Приближенное решение
def S(x, nodes, b_coeff):
    result = 0.0
    for i in range(-1, len(b_coeff) - 1):
        result += b_coeff[i + 1] * B_spline(x, i, nodes)
    return result


# Проверка и вывод результатов
table = PrettyTable()
table.field_names = ["x", "Приближенное", "Точное", "Ошибка"]

x_vals = np.linspace(0, 1, 6)
for x in x_vals:
    approx = S(x, nodes, b_coeff)
    exact = x / (x + 1)
    table.add_row([f"{x:.2f}", f"{approx:.6f}", f"{exact:.6f}", f"{abs(approx - exact):.2e}"])

print("Результаты:")
print(table)