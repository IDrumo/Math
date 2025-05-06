import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from prettytable import PrettyTable

# Параметры задачи (вариант 19)
alpha0, beta0, gamma0 = 0, 1, 1
alpha1, beta1, gamma1 = 0, 1, 1 / (np.cos(1)**2)
exact_solution = lambda x: np.tan(x) - 1
p = lambda x: -1
q = lambda x: 0
f = lambda x: (2 * np.sin(x) / np.cos(x)**3) - (1 / np.cos(x)**2)

# Параметры сетки
N = 100
h = 1.0 / N
x = np.linspace(0, 1, N + 1)

# Инициализация матрицы в ЛЕНТОЧНОМ формате (3 строки)
A = np.zeros((3, N + 1))
F = np.zeros(N + 1)

# Заполнение коэффициентов для внутренних точек (i=1..N-1)
for i in range(1, N):
    r = p(x[i]) / 2 * h
    abs_r = abs(r)
    th = np.tanh(abs_r)

    a_i = (1 + abs_r - th - r) / h**2
    c_i = (1 + abs_r - th + r) / h**2
    b_i = q(x[i]) - a_i - c_i

    # Заполнение ленты:
    A[0, i+1] = c_i    # Верхняя диагональ (c_i)
    A[1, i] = b_i      # Главная диагональ (b_i)
    A[2, i-1] = a_i    # Нижняя диагональ (a_i)
    F[i] = f(x[i])

# Левое граничное условие (i=0)
# Аппроксимация u'(0) = (-3y0 + 4y1 - y2)/(2h) = gamma0
A[1, 0] = -3/(2*h)     # Главная диагональ
A[0, 1] = 4/(2*h)      # Верхняя диагональ
A[2, 0] = -1/(2*h)     # Нижняя диагональ (y2)
F[0] = gamma0

# Правое граничное условие (i=N)
# Аппроксимация u'(1) = (3yN - 4yN-1 + yN-2)/(2h) = gamma1
A[1, N] = 3/(2*h)      # Главная диагональ
A[2, N-1] = -4/(2*h)   # Нижняя диагональ (yN-1)
A[2, N-2] = 1/(2*h)    # Нижняя диагональ (yN-2)
F[N] = gamma1

# Решение системы методом прогонки (БЕЗ обрезки матрицы!)
solution = solve_banded((1, 1), A, F)  # Вся матрица

# Граничные значения уже учтены в решении
y = solution

# Вычисление точного решения
exact = exact_solution(x)

# Вычисление погрешности
error = np.max(np.abs(y - exact))
print(f"Максимальная погрешность: {error:.6e}")

# Вывод результатов в таблице (сокращенный вариант)
table = PrettyTable()
table.field_names = ["x", "Точное значение", "Численное значение", "Ошибка"]
for i in range(0, N+1, 10):  # Каждая 10-я точка
    table.add_row([f"{x[i]:.3f}", f"{exact[i]:.6f}", f"{y[i]:.6f}", f"{abs(y[i]-exact[i]):.2e}"])
print(table)