from typing import Callable

import numpy as np
from numpy import ndarray
from prettytable import PrettyTable


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


class DifferentialEquationSolver:
    def fdm(self,
            alpha_0: float,
            alpha_1: float,
            beta_0: float,
            beta_1: float,
            gamma_0: float,
            gamma_1: float,
            n: int,
            h: float,
            x: ndarray,
            p_x: Callable[[float | ndarray], float | ndarray],
            q_x: Callable[[float | ndarray], float | ndarray],
            f_x: Callable[[float | ndarray], float | ndarray],
            a_hr: Callable[[float | ndarray, float | ndarray], float | ndarray],
            c_hr: Callable[[float | ndarray, float | ndarray], float | ndarray]) -> ndarray:
        r = p_x(x) * h / 2
        a_values = a_hr(h, r)  # Теперь возвращает массив
        c_values = c_hr(h, r)  # Теперь возвращает массив
        b_values = q_x(x) - a_values - c_values
        f_values = f_x(x)

        A = np.zeros((n + 1, n + 1))
        B = np.zeros(n + 1)

        # Заполнение внутренних точек (i=1..n-1)
        for i in range(1, n):
            A[i][i - 1] = a_values[i]
            A[i][i] = b_values[i]
            A[i][i + 1] = c_values[i]
            B[i] = f_values[i]

        # Левое граничное условие
        A[0, 0] = -3 / (2 * h)
        A[0, 1] = 4 / (2 * h)
        B[0] = gamma_0

        # Правое граничное условие
        A[n, n - 1] = -4 / (2 * h)
        A[n, n] = 3 / (2 * h)
        A[n, n - 2] = 1 / (2 * h)
        B[n] = gamma_1

        solver = AlgebraicSolver()
        y = solver.tdma(A, B)
        return y


def p(x: float | ndarray) -> float | ndarray:
    return -1 * np.ones_like(x)  # Возвращает массив


def q(x: float | ndarray) -> float | ndarray:
    return np.zeros_like(x)  # Возвращает массив


def f(x: float | ndarray) -> float | ndarray:
    return (2 * np.sin(x) / np.cos(x) ** 3) - (1 / np.cos(x) ** 2)


def true_function(x: float | ndarray) -> float | ndarray:
    return np.tan(x) - 1


def a(h: float | ndarray, r: float | ndarray) -> float | ndarray:
    return (1 + np.abs(r) - np.tanh(np.abs(r)) - r) / (h ** 2)


def c(h: float | ndarray, r: float | ndarray) -> float | ndarray:
    return (1 + np.abs(r) - np.tanh(np.abs(r)) + r) / (h ** 2)


def main():
    alpha_0: float = 0
    alpha_1: float = 0
    beta_0: float = 1
    beta_1: float = 1
    gamma_0: float = 1
    gamma_1: float = 1 / (np.cos(1) ** 2)
    n: int = 100
    h: float = 1 / n
    x: ndarray = np.linspace(0, 1, n + 1)

    dif_solver = DifferentialEquationSolver()
    y = dif_solver.fdm(alpha_0, alpha_1, beta_0, beta_1, gamma_0, gamma_1, n, h, x, p, q, f, a, c)

    y_true = true_function(x)
    errors = y - y_true

    table = PrettyTable()
    table.field_names = ["x", "Истинное значение", "Трехточечный метод", "Ошибка"]
    for i in range(0, len(x), 10):  # Вывод каждую 10-ю точку
        table.add_row([f"{x[i]:.4f}", f"{y_true[i]:.6f}", f"{y[i]:.6f}", f"{errors[i]:.2e}"])
    print(table)


if __name__ == '__main__':
    main()
