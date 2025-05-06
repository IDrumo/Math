import numpy as np
import matplotlib.pyplot as plt

# Пространственная и временная сетка
L = 1.0
h = 0.1
tau = 0.001
T = 0.05

x = np.arange(0, L + h, h)
y = np.arange(0, L + h, h)
X, Y = np.meshgrid(x, y)

Nx, Ny = len(x), len(y)
Nt = int(T / tau)

# Точное решение
def u_exact(t, x, y):
    return t * np.sin(np.pi * x) * np.sin(np.pi * y)

# Правая часть уравнения
def f_rhs(t, x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * (1 + 2 * np.pi**2 * t)

# Начальное условие
v = np.zeros((Nx, Ny))
v_half = np.zeros_like(v)
v_new = np.zeros_like(v)

# Оператор Лапласа
def laplacian(u, h):
    lap = np.zeros_like(u)
    lap[1:-1,1:-1] = (
        u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2] - 4 * u[1:-1,1:-1]
    ) / h**2
    return lap

# Временной цикл
for n in range(Nt):
    t_n = n * tau
    t_half = t_n + 0.5 * tau
    t_np1 = t_n + tau

    # Шаг 1: на полушаге
    lap_v = laplacian(v, h)
    v_half[1:-1,1:-1] = v[1:-1,1:-1] + tau * (
        lap_v[1:-1,1:-1] + f_rhs(t_n, X[1:-1,1:-1], Y[1:-1,1:-1])
    )

    # Граничные условия (v = 0 на границе)
    v_half[0,:] = 0
    v_half[-1,:] = 0
    v_half[:,0] = 0
    v_half[:,-1] = 0

    # Шаг 2: полный шаг
    lap_v_half = laplacian(v_half, h)
    v_new[1:-1,1:-1] = v_half[1:-1,1:-1] + tau * (
        lap_v_half[1:-1,1:-1] + f_rhs(t_half, X[1:-1,1:-1], Y[1:-1,1:-1])
    )

    v_new[0,:] = 0
    v_new[-1,:] = 0
    v_new[:,0] = 0
    v_new[:,-1] = 0

    v = v_new.copy()

# Точное решение на финальный момент времени
u_true = u_exact(T, X, Y)

# Визуализация численного и точного решений
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Стиль для визуализации
cmap = 'viridis'

# Численное решение
contour = axs[0].pcolormesh(X, Y, v, cmap=cmap, shading='auto')
plt.colorbar(contour, ax=axs[0])
axs[0].set_title('Численное решение')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

# Точное решение
contour = axs[1].pcolormesh(X, Y, u_true, cmap=cmap, shading='auto')
plt.colorbar(contour, ax=axs[1])
axs[1].set_title('Точное решение')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

# Абсолютная ошибка
contour = axs[2].pcolormesh(X, Y, np.abs(u_true - v), cmap=cmap, shading='auto')
plt.colorbar(contour, ax=axs[2])
axs[2].set_title('Абсолютная ошибка')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')

plt.tight_layout()
plt.show()