import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Параметры задачи
L = 1.0
N = 10
h = L / N
tau = 0.0001  # Шаг по времени (подбирается для устойчивости)
T_final = 0.1
Nt = int(T_final / tau)
x = np.linspace(0, L, N + 1)
y = np.linspace(0, L, N + 1)
X, Y = np.meshgrid(x, y)


# Точное решение для проверки
def exact_solution(t):
    return t * np.sin(np.pi * X) * np.sin(np.pi * Y)


# Функция источника f(x, y, t)
def f_func(t):
    return (1 + 2 * t * np.pi ** 2) * np.sin(np.pi * X) * np.sin(np.pi * Y)


# Инициализация решения
u = np.zeros((N + 1, N + 1))

# Коэффициенты для неявных схем
alpha = tau / (2 * h ** 2)


# Матрицы для неявных шагов по x и y
def create_matrix(alpha, size):
    diagonals = [-alpha * np.ones(size - 1), (1 + 2 * alpha) * np.ones(size), -alpha * np.ones(size - 1)]
    return diags(diagonals, [-1, 0, 1], format='csc')


A_x = create_matrix(alpha, N + 1)
A_y = create_matrix(alpha, N + 1)

# Метод расщепления (ADI)
for n in range(Nt):
    t = n * tau

    # Первый полушаг: неявно по x, явно по y
    rhs = u.copy()
    # Добавляем вклад от y-производной и источника
    rhs += tau / 2 * (np.roll(u, 1, axis=0) - 2 * u + np.roll(u, -1, axis=0)) / h ** 2
    rhs += tau / 2 * f_func(t)

    # Решаем систему по x
    for j in range(1, N):
        u_half = spsolve(A_x, rhs[1:-1, j])
        u[1:-1, j] = u_half

    # Второй полушаг: неявно по y, явно по x
    rhs = u.copy()
    # Добавляем вклад от x-производной и источника
    rhs += tau / 2 * (np.roll(u, 1, axis=1) - 2 * u + np.roll(u, -1, axis=1)) / h ** 2
    rhs += tau / 2 * f_func(t + tau / 2)

    # Решаем систему по y
    for i in range(1, N):
        u_next = spsolve(A_y, rhs[i, 1:-1])
        u[i, 1:-1] = u_next

# Проверка устойчивости: сравнение с точным решением
t_final = T_final
u_exact = exact_solution(t_final)
error = np.abs(u - u_exact).max()
print(f"Максимальная ошибка: {error:.6f}")

# Визуализация
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.contourf(X, Y, u, levels=20, cmap='viridis')
plt.colorbar()
plt.title('Численное решение')

plt.subplot(132)
plt.contourf(X, Y, u_exact, levels=20, cmap='viridis')
plt.colorbar()
plt.title('Точное решение')

plt.subplot(133)
plt.contourf(X, Y, error, levels=20, cmap='hot')
plt.colorbar()
plt.title('Ошибка')
plt.show()