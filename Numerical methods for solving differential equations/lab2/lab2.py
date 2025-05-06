import numpy as np
from prettytable import PrettyTable

# Параметры задачи (вариант 19)
alpha0, beta0, gamma0 = 0, 1, 1
alpha1, beta1, gamma1 = 0, 1, 1 / (np.cos(1)**2)
p = lambda x: -1
q = lambda x: 0
f = lambda x: (2 * np.sin(x) / np.cos(x)**3) - (1 / np.cos(x)**2)

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
    th = np.tanh(abs_r)

    a_i = (1 + abs_r - th - r) / h**2
    c_i = (1 + abs_r - th + r) / h**2
    b_i = q(x[i]) - a_i - c_i

    # Заполнение ленты
    A[0, i+1] = c_i    # Верхняя диагональ (c_i)
    A[1, i] = b_i      # Главная диагональ (b_i)
    A[2, i-1] = a_i    # Нижняя диагональ (a_i)
    F[i] = f(x[i])

# Левое граничное условие (i=0)
A[1, 0] = -3/(2*h)     # Главная диагональ
A[0, 1] = 4/(2*h)      # Верхняя диагональ
A[2, 0] = -1/(2*h)     # Нижняя диагональ (y2)
F[0] = gamma0

# Правое граничное условие (i=N)
A[1, N] = 3/(2*h)      # Главная диагональ
A[2, N-1] = -4/(2*h)   # Нижняя диагональ (yN-1)
A[2, N-2] = 1/(2*h)    # Нижняя диагональ (yN-2)
F[N] = gamma1

# Решение системы методом прогонки
def tdma(A, F):
    n = len(F)
    a = [0] + [A[2, i] for i in range(1, n)]
    b = [A[1, i] for i in range(n)]
    c = [A[0, i] for i in range(n - 1)] + [0]
    F = list(map(float, F))

    alpha = [-c[0] / b[0]]
    beta = [F[0] / b[0]]
    x = np.zeros(n)
    for i in range(1, n):
        alpha.append(-c[i] / (a[i] * alpha[i - 1] + b[i]))
        beta.append((F[i] - a[i] * beta[i - 1]) / (a[i] * alpha[i - 1] + b[i]))

    x[n - 1] = beta[n - 1]
    for i in range(n - 1, 0, -1):
        x[i - 1] = alpha[i - 1] * x[i] + beta[i - 1]

    return x

y = tdma(A, F)

# Точное решение (исправленное)
def exact_solution(x):
    return np.tan(x) - 1

# Вычисление погрешности
error = np.max(np.abs(y - exact_solution(x)))
print(f"Максимальная погрешность: {error:.2e}")

# Вывод таблицы
table = PrettyTable()
table.field_names = ["x", "Точное", "Численное", "Ошибка"]
for i in range(0, N+1, 10):
    table.add_row([f"{x[i]:.2f}", f"{exact_solution(x[i]):.6f}", f"{y[i]:.6f}", f"{abs(y[i]-exact_solution(x[i])):.2e}"])
print(table)