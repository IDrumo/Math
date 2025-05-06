import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from prettytable import PrettyTable

# Параметры задачи (вариант 3)
alpha0, beta0, gamma0 = 0, 1, 0
alpha1, beta1, gamma1 = 1, 0, 0.5
exact_solution = lambda x: 1 / (x ** 2 + 1)
p = lambda x: 4 * x / (x ** 2 + 1)
q = lambda x: 1 / (x ** 2 + 1)
f = lambda x: 3 / (x ** 2 + 1) ** 2

# Параметры сетки
N = 100
h = 1.0 / N
x = np.linspace(0, 1, N + 1)

# Инициализация матрицы в ленточном формате (3 строки)
A = np.zeros((3, N + 1))
F = np.zeros(N + 1)

# Заполнение коэффициентов для внутренних точек (i=1..N-1)
for i in range(1, N):
    r = p(x[i]) / 2 * h
    abs_r = np.abs(r)

    # Схема Самарского
    a_i = (1 + (r ** 2) / (1 + abs_r) - r) / h ** 2
    c_i = (1 + (r ** 2) / (1 + abs_r) + r) / h ** 2
    b_i = q(x[i]) - a_i - c_i

    # Заполнение ленты
    A[0, i + 1] = c_i  # Верхняя диагональ (c_i)
    A[1, i] = b_i  # Главная диагональ (b_i)
    A[2, i - 1] = a_i  # Нижняя диагональ (a_i)
    F[i] = f(x[i])

# Левое граничное условие: beta0*u'(0) + alpha0*u(0) = gamma0
A[1, 0] = -3 / (2 * h)  # Главная диагональ (y0)
A[0, 1] = 4 / (2 * h)  # Верхняя диагональ (y1)
A[2, 0] = -1 / (2 * h)  # Нижняя диагональ (y2)
F[0] = gamma0

# Правое граничное условие: alpha1*u(1) = gamma1
A[1, N] = 1  # Главная диагональ (yN)
A[0, N] = 0  # Верхняя диагональ (не используется)
A[2, N] = 0  # Нижняя диагональ (не используется)
F[N] = gamma1

# Решение системы
y = solve_banded((1, 1), A, F)

# Точное решение и погрешность
exact = exact_solution(x)
error = np.max(np.abs(y - exact))
print(f"Максимальная погрешность: {error:.2e}")

# Вывод таблицы
table = PrettyTable()
table.field_names = ["x", "Точное", "Численное", "Ошибка"]
for i in range(0, N + 1, 10):
    table.add_row([f"{x[i]:.2f}", f"{exact[i]:.6f}", f"{y[i]:.6f}", f"{abs(y[i] - exact[i]):.2e}"])
print(table)

# График
plt.plot(x, y, label="Численное")
plt.plot(x, exact, "--", label="Точное")
plt.xlabel("x"), plt.ylabel("u(x)"), plt.legend(), plt.grid()
plt.show()