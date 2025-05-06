import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
a = 1.0  # предполагаем a=1, так как не указано в задании
I = 1.0
T = 1.0
M = 10
N = 10
h = I / M
tau = T / N

# Создание сетки
x = np.linspace(0, I, M+1)
t = np.linspace(0, T, N+1)
u = np.zeros((M+1, N+1))

# Начальные условия
def phi(x):
    return 3 * x * (1 - x**2)

def psi(x):
    return x**3 + x**2

def g(x):
    return 4 * x * (x**2 - 1)

# Заполнение начального условия u(x, 0)
for m in range(M+1):
    u[m, 0] = phi(x[m])

# Граничные условия (γ0(t) = 0, γ1(t) = 0)
u[0, :] = 0.0
u[M, :] = 0.0

# Вычисление первого временного слоя (n=1)
for m in range(1, M):
    term1 = u[m, 0] + tau * psi(x[m])
    laplacian = (u[m-1, 0] - 2*u[m, 0] + u[m+1, 0]) / h**2
    term2 = (tau**2 / 2) * (a**2 * laplacian + g(x[m]))
    u[m, 1] = term1 + term2

# Вычисление последующих временных слоев
for n in range(1, N):
    for m in range(1, M):
        laplacian = (u[m-1, n] - 2*u[m, n] + u[m+1, n]) / h**2
        u[m, n+1] = 2*u[m, n] - u[m, n-1] + (a**2 * tau**2) * laplacian + tau**2 * g(x[m])

# Визуализация результатов
plt.figure(figsize=(10, 6))
for n in range(0, N+1, 2):
    plt.plot(x, u[:, n], label=f't={t[n]:.2f}')
plt.title('Решение уравнения колебаний струны')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.grid(True)
plt.show()