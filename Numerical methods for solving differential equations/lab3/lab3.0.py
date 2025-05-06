import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt


class AlgebraicSolver:

    def tdma(self, matrix: ndarray, f: ndarray) -> ndarray:
        n = len(f)
        a = [0] + [matrix[i][i - 1] for i in range(1, n)]
        b = [matrix[i][i] for i in range(n)]
        c = [matrix[i][i + 1] for i in range(n - 1)] + [0]
        f = list(map(float, f))

        alpha = [-c[0] / b[0]]
        beta = [f[0] / b[0]]
        x: ndarray = np.zeros(n)
        for i in range(1, n):
            alpha.append(-c[i] / (a[i] * alpha[i - 1] + b[i]))
            beta.append((f[i] - a[i] * beta[i - 1]) / (a[i] * alpha[i - 1] + b[i]))

        x[n - 1] = beta[n - 1]

        for i in range(n - 1, 0, -1):
            x[i - 1] = alpha[i - 1] * x[i] + beta[i - 1]

        return x


def phi(x, t):
    return x ** 2 * t * (1 - x)


def psi(x):
    return x * (x - 1)  # Исправлен оператор ^ на **


def explicit_scheme(a, gamma_0, gamma_1, M, N, l, T):
    h = l / M
    tau = T / N
    lambda_ = (a ** 2 * tau) / (h ** 2)
    print(lambda_)
    if lambda_ > 0.5:
        print(
            f"Warning: Explicit scheme may be unstable (lambda={lambda_:.3f} > 0.5). Consider reducing tau or increasing M.")

    x = np.linspace(0, l, M + 1)
    t = np.linspace(0, T, N + 1)
    u = np.zeros((N + 1, M + 1))

    # Начальное условие
    for m in range(M + 1):
        u[0, m] = psi(x[m])

    # Граничные условия
    for n in range(N + 1):
        u[n, 0] = gamma_0
        u[n, M] = gamma_1

    # Явная разностная схема
    for n in range(N):
        for m in range(1, M):
            u[n + 1, m] = (
                    u[n, m] + lambda_ * (u[n, m + 1] - 2 * u[n, m] + u[n, m - 1]) + tau * phi(x[m], t[n])
            )

    return x, t, u


def implicit_scheme(a, gamma_0, gamma_1, M, N, l, T):
    h = l / M
    tau = T / N
    lambda_ = (a ** 2 * tau) / (h ** 2)

    x = np.linspace(0, l, M + 1)
    t = np.linspace(0, T, N + 1)
    u = np.zeros((N + 1, M + 1))

    # Начальное условие
    for m in range(M + 1):
        u[0, m] = psi(x[m])

    # Граничные условия
    for n in range(N + 1):
        u[n, 0] = gamma_0
        u[n, M] = gamma_1

    # Неявная схема (метод прогонки)
    A = -lambda_ * np.ones(M - 1)
    B = (1 + 2 * lambda_) * np.ones(M - 1)
    C = -lambda_ * np.ones(M - 1)

    for n in range(N):
        d = u[n, 1:M] + tau * phi(x[1:M], t[n])
        d[0] += lambda_ * gamma_0
        d[-1] += lambda_ * gamma_1
        u[n + 1, 1:M] = initiate_tdma(A, B, C, d)

    return x, t, u


def initiate_tdma(A, B, C, d):
    solver = AlgebraicSolver()
    n = len(B)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, B)
    np.fill_diagonal(matrix[1:], A)
    np.fill_diagonal(matrix[:, 1:], C)
    return solver.tdma(matrix, d)


def plot_solution(x, t, u, title):
    X, T = np.meshgrid(x, t)
    plt.figure(figsize=(8, 6))
    plt.contourf(X, T, u, 20, cmap="hot")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    plt.show()


def main():
    a = 0.001
    gamma_0 = gamma_1 = 0
    M = 1000
    N = 6
    l = 1
    T = 2

    x, t, u_explicit = explicit_scheme(a, gamma_0, gamma_1, M, N, l, T)
    plot_solution(x, t, u_explicit, "Heat Equation Solution (Explicit Scheme)")

    x, t, u_implicit = implicit_scheme(a, gamma_0, gamma_1, M, N, l, T)
    plot_solution(x, t, u_implicit, "Heat Equation Solution (Implicit Scheme)")


if __name__ == "__main__":
    main()
