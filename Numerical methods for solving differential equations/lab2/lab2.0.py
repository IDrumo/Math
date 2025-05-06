from typing import Callable

import numpy as np
from numpy import ndarray
from prettytable import PrettyTable


class DifferentialEquationSolver:
    """
    A class for solving differential equations using different methods.

    Methods:
        fdm(...): Constructs the finite difference scheme from given differential operators and boundary conditions
                  and solves the resulting linear system.
    """

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
        """
        Solves a differential equation using the finite difference method by constructing and solving
        the linear system A*y = B based on the discretized equation and boundary conditions.

        Args:
            alpha_0 (float): Coefficient for the left boundary condition.
            alpha_1 (float): Coefficient for the right boundary condition.
            beta_0 (float): Coefficient for the left boundary condition.
            beta_1 (float): Coefficient for the right boundary condition.
            gamma_0 (float): Constant for the left boundary condition.
            gamma_1 (float): Constant for the right boundary condition.
            n (int): Number of subintervals.
            h (float): Step size.
            x (ndarray): Array of x values.
            p_x (Callable): Function to compute p(x).
            q_x (Callable): Function to compute q(x).
            f_x (Callable): Function to compute f(x).
            a_hr (Callable): Function to compute the finite difference coefficient a.
            c_hr (Callable): Function to compute the finite difference coefficient c.

        Returns:
            ndarray: Solution vector y for the discretized differential equation.
        """
        r = p_x(x) * h / 2
        a_values = a_hr(h, r)
        c_values = c_hr(h, r)
        b_values = q_x(x) - a_values - c_values
        f_values = f_x(x)

        A = np.zeros((n + 1, n + 1))
        B = np.zeros(n + 1)

        for i in range(1, n):
            A[i][i - 1] = a_values[i]
            A[i][i] = b_values[i]
            A[i][i + 1] = c_values[i]
            B[i] = f_values[i]

        A[0, 0] = alpha_0 - beta_0 / h
        A[0, 1] = beta_0 / h
        B[0] = gamma_0

        A[n, n - 1] = -beta_1 / h
        A[n, n] = alpha_1 + beta_1 / h
        B[n] = gamma_1

        solver = AlgebraicSolver()
        y = solver.tdma(A, B)

        return y

    def runge_kutt_4(self,
                     x: ndarray,
                     y0: float,
                     h: float,
                     n: int,
                     f: Callable[[float, float, float, float, float], float],
                     a: float,
                     b: float,
                     c: float) -> ndarray:
        """
            Solves a differential equation using the 4th order Runge-Kutta method.

            Args:
                x (ndarray): Array of x values.
                y0 (float): Initial value of y.
                h (float): Step size.
                n (int): Number of steps.
                f (Callable): Function representing the differential equation.
                a (float): Parameter a of the equation.
                b (float): Parameter b of the equation.
                c (float): Parameter c of the equation.

            Returns:
                ndarray: Array of y values computed using the Runge-Kutta method.
            """
        y_values = np.array([y0], dtype=float)

        for i in range(n):
            x_current = float(x[i])
            y_current = float(y_values[-1])

            k1 = f(x_current, y_current, a, b, c)
            k2 = f(x_current + h / 2, y_current + h / 2 * k1, a, b, c)
            k3 = f(x_current + h / 2, y_current + h / 2 * k2, a, b, c)
            k4 = f(x_current + h, y_current + h * k3, a, b, c)

            y_values = np.append(y_values, y_current + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4))

        return y_values


class AlgebraicSolver:
    """
    A class to solve algebraic equations using the Thomas algorithm (TDMA).

    Methods:
    --------
    tdma(matrix: ndarray, f: ndarray) -> ndarray
        Solves a tridiagonal system of linear equations.
    """

    def tdma(self, matrix: ndarray, f: ndarray) -> ndarray:
        """
        Solves a tridiagonal system of linear equations using the Thomas algorithm.

        Parameters:
        -----------
        matrix : ndarray
            Coefficient matrix of the system.
        f : ndarray
            Right-hand side vector.

        Returns:
        --------
        ndarray
            Solution vector.
        """
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


def p(x: float | ndarray) -> float | ndarray:
    """
    Compute the value of p(x).

    Args:
        x (float | ndarray): Input value or array.

    Returns:
        float | ndarray: The computed p(x) = 4*x / (x^2 + 1).
    """
    return 4 * x / (x ** 2 + 1)


def q(x: float | ndarray) -> float | ndarray:
    """
    Compute the value of q(x).

    Args:
        x (float | ndarray): Input value or array.

    Returns:
        float | ndarray: The computed q(x) = -1 / (x^2 + 1).
    """
    return -1 / (x ** 2 + 1)


def f(x: float | ndarray) -> float | ndarray:
    """
    Compute the value of f(x).

    Args:
        x (float | ndarray): Input value or array.

    Returns:
        float | ndarray: The computed f(x) = -3 / ((x^2 + 1)^2).
    """
    return -3 / ((x ** 2 + 1) ** 2)


def true_function(x: float | ndarray) -> float | ndarray:
    """
    Compute the true analytical solution.

    Args:
        x (float | ndarray): Input value or array.

    Returns:
        float | ndarray: The true function value 1 / (x^2 + 1).
    """
    return 1 / (x ** 2 + 1)


def a(h: float | ndarray, r: float | ndarray) -> float | ndarray:
    """
    Compute the coefficient a for the finite difference scheme.

    Args:
        h (float | ndarray): Step size.
        r (float | ndarray): Parameter computed from p(x) and h.

    Returns:
        float | ndarray: The computed coefficient a.
    """
    return (1 + (r ** 2) / (1 + np.abs(r)) - r) / (h ** 2)


def c(h: float | ndarray, r: float | ndarray) -> float | ndarray:
    """
    Compute the coefficient c for the finite difference scheme.

    Args:
        h (float | ndarray): Step size.
        r (float | ndarray): Parameter computed from p(x) and h.

    Returns:
        float | ndarray: The computed coefficient c.
    """
    return (1 + (r ** 2) / (1 + np.abs(r)) + r) / (h ** 2)


def main():
    """
    Main function to solve the boundary value problem using the finite difference method.

    The function defines the differential equation parameters, calls the solver,
    computes the true solution, calculates errors, and prints the results in a table.
    """
    alpha_0: float = 0
    alpha_1: float = 1
    beta_0: float = 1
    beta_1: float = 0
    gama_0: float = 0
    gama_1: float = 0.5
    n: int = 100
    h: float = 1 / n
    x: ndarray = np.linspace(0, 1, n + 1)

    dif_solver = DifferentialEquationSolver()
    y = dif_solver.fdm(alpha_0, alpha_1, beta_0, beta_1, gama_0, gama_1, n, h, x, p, q, f, a, c)

    y_true = true_function(x)
    errors = y - y_true

    table = PrettyTable()
    table.field_names = ["x", "True Value", "Three-Point Method", "Error"]

    for i in range(len(x)):
        table.add_row([round(x[i], 5), round(y_true[i], 5), round(y[i], 5), round(errors[i], 5)])

    print(table)


if __name__ == '__main__':
    main()
