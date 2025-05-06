import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML

# Параметры задачи
L = 1.0
h = 0.1
tau = 0.001
T = 0.05
save_interval = 5  # Сохранять каждый 5-й кадр для анимации

# Сетка
x = np.arange(0, L + h, h)
y = np.arange(0, L + h, h)
X, Y = np.meshgrid(x, y)
Nx, Ny = len(x), len(y)
Nt = int(T / tau)


# Аналитические функции
def u_exact(t, x, y):
    return t * np.sin(np.pi * x) * np.sin(np.pi * y)


def f_rhs(t, x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * (1 + 2 * np.pi ** 2 * t)


# Инициализация
v = np.zeros((Nx, Ny))
v_half = np.zeros_like(v)
v_new = np.zeros_like(v)
solutions = []  # Для хранения кадров анимации


# Оптимизированный оператор Лапласа
def laplacian(u, h):
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / h ** 2
    return lap


# Временной цикл с сохранением кадров
for n in range(Nt):
    t_n = n * tau

    # Шаг 1: Полушаг
    lap_v = laplacian(v, h)
    v_half[1:-1, 1:-1] = v[1:-1, 1:-1] + tau * (lap_v[1:-1, 1:-1] + f_rhs(t_n, X[1:-1, 1:-1], Y[1:-1, 1:-1]))
    v_half[[0, -1], :] = v_half[:, [0, -1]] = 0  # Граничные условия

    # Шаг 2: Полный шаг
    lap_v_half = laplacian(v_half, h)
    v_new[1:-1, 1:-1] = v_half[1:-1, 1:-1] + tau * (
                lap_v_half[1:-1, 1:-1] + f_rhs(t_n + 0.5 * tau, X[1:-1, 1:-1], Y[1:-1, 1:-1]))
    v_new[[0, -1], :] = v_new[:, [0, -1]] = 0

    v = v_new.copy()

    if n % save_interval == 0:
        solutions.append((t_n, v.copy()))

# Визуализация
plt.style.use('seaborn-v0_8')

# 1. Сравнение решений в финальный момент
u_true = u_exact(T, X, Y)
error = np.abs(u_true - v)

fig = plt.figure(figsize=(18, 6))

# 3D визуализация
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X, Y, v, cmap='viridis', rstride=1, cstride=1)
ax1.set_title(f'Численное решение (t={T:.3f})')
ax1.set_zlim(0, T)

ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X, Y, u_true, cmap='plasma', rstride=1, cstride=1)
ax2.set_title('Точное решение')
ax2.set_zlim(0, T)

ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X, Y, error, cmap='hot', rstride=1, cstride=1)
ax3.set_title('Абсолютная ошибка')
fig.colorbar(surf3, ax=ax3, shrink=0.5)

# 2. Анимация эволюции решения
fig_anim = plt.figure(figsize=(10, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')


def update_anim(frame):
    ax_anim.clear()
    t, data = solutions[frame]
    surf = ax_anim.plot_surface(X, Y, data, cmap='viridis')
    ax_anim.set_title(f'Распределение температуры (t={t:.3f})')
    ax_anim.set_zlim(0, T)
    return surf,


ani = FuncAnimation(fig_anim, update_anim, frames=len(solutions), interval=100, blit=False)
plt.close(fig_anim)

# 3. График ошибки по времени
errors = [np.max(np.abs(u_exact(t, X, Y) - sol)) for t, sol in solutions]
times = [t for t, sol in solutions]

fig_err = plt.figure(figsize=(8, 5))
plt.plot(times, errors, 'r-o', linewidth=2, markersize=5)
plt.title('Максимальная ошибка по времени')
plt.xlabel('Время')
plt.ylabel('Максимальная ошибка')
plt.grid(True)

plt.tight_layout()
plt.show()

# Для отображения анимации в Jupyter Notebook
HTML(ani.to_html5_video())