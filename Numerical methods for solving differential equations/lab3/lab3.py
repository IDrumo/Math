import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Параметры задачи
a = 1.0  # Коэффициент теплопроводности
l = 1.0  # Длина пространственного интервала
T = 1.0  # Временной интервал

# Параметры сетки
M = 10  # Количество пространственных узлов
N = 10000  # Количество временных шагов
h = l / M
tau = T / N

sigma = a ** 2 * tau / h ** 2
if sigma > 0.5:
    print(f"Предупреждение: sigma = {sigma:.3f} > 0.5, явный метод неустойчив!")


# Вариант 7.1
def phi1(x, t):
    return x ** 2 * t * (1 - x)


def psi1(x):
    return x * (1 - x)


def gamma0_1(t):
    return 0.0


def gamma1_1(t):
    return 0.0


# Вариант 7.2
def phi2(x, t):
    return 0.0


def psi2(x):
    return 0.0


def gamma0_2(t):
    return -t * (t ** 2 + 1)


def gamma1_2(t):
    return t ** 2


# Явный метод
def explicit_method(phi, psi, gamma0, gamma1):
    u = np.zeros((M + 1, N + 1))
    x = np.linspace(0, l, M + 1)
    t = np.linspace(0, T, N + 1)

    # Начальные условия
    u[:, 0] = psi(x)

    # Граничные условия
    for n in range(N + 1):
        u[0, n] = gamma0(t[n])
        u[-1, n] = gamma1(t[n])

    # Явная схема
    sigma = a ** 2 * tau / h ** 2
    for n in range(N):
        for m in range(1, M):
            u[m, n + 1] = u[m, n] + sigma * (u[m + 1, n] - 2 * u[m, n] + u[m - 1, n]) + tau * phi(x[m], t[n])

    return u, x, t


# Чисто неявная схема
def implicit_method(phi, psi, gamma0, gamma1):
    u = np.zeros((M + 1, N + 1))
    x = np.linspace(0, l, M + 1)
    t = np.linspace(0, T, N + 1)

    # Начальные условия
    u[:, 0] = psi(x)

    # Граничные условия
    for n in range(N + 1):
        u[0, n] = gamma0(t[n])
        u[-1, n] = gamma1(t[n])

    # Коэффициенты для метода прогонки
    sigma = a ** 2 * tau / h ** 2
    alpha = np.zeros(M - 1)
    beta = np.zeros(M - 1)

    for n in range(N):
        A = np.zeros(M - 1)
        B = np.zeros(M - 1)
        C = np.zeros(M - 1)
        D = np.zeros(M - 1)

        for m in range(1, M):
            A[m - 1] = -sigma
            B[m - 1] = 1 + 2 * sigma
            C[m - 1] = -sigma
            D[m - 1] = u[m, n] + tau * phi(x[m], t[n])

        # Прогонка
        alpha[0] = C[0] / B[0]
        beta[0] = D[0] / B[0]

        for i in range(1, M - 1):
            alpha[i] = C[i] / (B[i] - A[i] * alpha[i - 1])
            beta[i] = (D[i] - A[i] * beta[i - 1]) / (B[i] - A[i] * alpha[i - 1])

        u[M - 1, n + 1] = (D[-1] - A[-1] * beta[-2]) / (B[-1] - A[-1] * alpha[-2])

        for i in range(M - 2, 0, -1):
            u[i, n + 1] = alpha[i] * u[i + 1, n + 1] + beta[i]

        u[0, n + 1] = gamma0(t[n + 1])
        u[-1, n + 1] = gamma1(t[n + 1])

    return u, x, t


# Схема Кранка-Николсона
def crank_nicolson_method(phi, psi, gamma0, gamma1):
    u = np.zeros((M + 1, N + 1))
    x = np.linspace(0, l, M + 1)
    t = np.linspace(0, T, N + 1)

    # Начальные условия
    u[:, 0] = psi(x)

    # Граничные условия
    for n in range(N + 1):
        u[0, n] = gamma0(t[n])
        u[-1, n] = gamma1(t[n])

    # Коэффициенты для метода прогонки
    sigma = a ** 2 * tau / (2 * h ** 2)
    alpha = np.zeros(M - 1)
    beta = np.zeros(M - 1)

    for n in range(N):
        A = np.zeros(M - 1)
        B = np.zeros(M - 1)
        C = np.zeros(M - 1)
        D = np.zeros(M - 1)

        for m in range(1, M):
            A[m - 1] = -sigma
            B[m - 1] = 1 + 2 * sigma
            C[m - 1] = -sigma
            D[m - 1] = (1 - 2 * sigma) * u[m, n] + sigma * (u[m + 1, n] + u[m - 1, n]) + tau * phi(x[m], t[n])

        # Прогонка
        alpha[0] = C[0] / B[0]
        beta[0] = D[0] / B[0]

        for i in range(1, M - 1):
            alpha[i] = C[i] / (B[i] - A[i] * alpha[i - 1])
            beta[i] = (D[i] - A[i] * beta[i - 1]) / (B[i] - A[i] * alpha[i - 1])

        u[M - 1, n + 1] = (D[-1] - A[-1] * beta[-2]) / (B[-1] - A[-1] * alpha[-2])

        for i in range(M - 2, 0, -1):
            u[i, n + 1] = alpha[i] * u[i + 1, n + 1] + beta[i]

        u[0, n + 1] = gamma0(t[n + 1])
        u[-1, n + 1] = gamma1(t[n + 1])

    return u, x, t


# Решение для варианта 7.1
u_explicit_1, x_1, t_1 = explicit_method(phi1, psi1, gamma0_1, gamma1_1)
u_implicit_1, _, _ = implicit_method(phi1, psi1, gamma0_1, gamma1_1)
u_crank_1, _, _ = crank_nicolson_method(phi1, psi1, gamma0_1, gamma1_1)

# Построение графиков
plt.figure(figsize=(12, 6))

# Вариант 7.1
plt.subplot(1, 2, 1)
times = [0, int(N / 5), int(2 * N / 5), int(3 * N / 5), int(4 * N / 5), N]
cmap_blue = colormaps.get_cmap('Blues')
cmap_red = colormaps.get_cmap('Reds')

for i, n in enumerate(times):
    color_blue = cmap_blue(i / len(times))
    plt.plot(x_1, u_explicit_1[:, n], color=color_blue, label=f'Явный t={t_1[n]:.2f}' if i == 0 else "")
    color_red = cmap_red(i / len(times))
    plt.plot(x_1, u_implicit_1[:, n], color=color_red, linestyle='--',
             label=f'Неявный t={t_1[n]:.2f}' if i == 0 else "")
    plt.plot(x_1, u_crank_1[:, n], color=color_red, linestyle='-.',
             label=f'Кранк-Николсон t={t_1[n]:.2f}' if i == 0 else "")

plt.title('Вариант 7.1')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()
plt.grid(True)

# Вариант 7.2

# Решение для варианта 7.2
u_explicit_2, x_2, t_2 = explicit_method(phi2, psi2, gamma0_1, gamma1_1)
u_implicit_2, _, _ = implicit_method(phi1, psi1, gamma0_1, gamma1_1)
u_crank_2, _, _ = crank_nicolson_method(phi2, psi2, gamma0_1, gamma1_1)

plt.subplot(1, 2, 2)
for i, n in enumerate(times):
    color_blue = cmap_blue(i / len(times))
    plt.plot(x_2, u_explicit_2[:, n], color=color_blue, label=f'Явный t={t_2[n]:.2f}' if i == 0 else "")
    color_red = cmap_red(i / len(times))
    plt.plot(x_2, u_implicit_2[:, n], color=color_red, linestyle='--',
             label=f'Неявный t={t_2[n]:.2f}' if i == 0 else "")
    plt.plot(x_2, u_crank_2[:, n], color=color_red, linestyle='-.',
             label=f'Кранк-Николсон t={t_2[n]:.2f}' if i == 0 else "")

plt.title('Вариант 7.2')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()