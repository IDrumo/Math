import numpy as np
from scipy.integrate import quad


# Определение функций p, q, f
def p(x):
    return np.exp((x ** 2) / 2 + x)


def q(x):
    return p(x)


def F(x):
    return (-2) / ((x + 1) ** 3) + 1


def f(x):
    return p(x) * F(x)


# Базисные функции и их производные
def phi0(x):
    return -0.5 * x ** 2 + x


def phi0_prime(x):
    return -x + 1


def phi1(x):
    return (x - 1) * x ** 2


def phi1_prime(x):
    return 3 * x ** 2 - 2 * x


def phi2(x):
    return (x - 1) * x ** 3


def phi2_prime(x):
    return 4 * x ** 3 - 3 * x ** 2


# Вычисление матрицы A и вектора B
def compute_A_B():
    # Матрица A
    def A11_integrand(x):
        return 2 * p(x) * phi1_prime(x) ** 2 - 2 * q(x) * phi1(x) ** 2

    A11, _ = quad(A11_integrand, 0, 1)

    def A12_integrand(x):
        return 2 * p(x) * phi1_prime(x) * phi2_prime(x) - 2 * q(x) * phi1(x) * phi2(x)

    A12, _ = quad(A12_integrand, 0, 1)

    def A22_integrand(x):
        return 2 * p(x) * phi2_prime(x) ** 2 - 2 * q(x) * phi2(x) ** 2

    A22, _ = quad(A22_integrand, 0, 1)

    A = np.array([[A11, A12], [A12, A22]])

    # Вектор B
    def B1_integrand(x):
        return (2 * p(x) * phi0_prime(x) * phi1_prime(x)
                - 2 * q(x) * phi0(x) * phi1(x)
                + 2 * f(x) * phi1(x))

    B1, _ = quad(B1_integrand, 0, 1)

    def B2_integrand(x):
        return (2 * p(x) * phi0_prime(x) * phi2_prime(x)
                - 2 * q(x) * phi0(x) * phi2(x)
                + 2 * f(x) * phi2(x))

    B2, _ = quad(B2_integrand, 0, 1)

    B = np.array([-B1, -B2])  # Перенос в правую часть уравнения

    return A, B


# Решение системы уравнений
A, B = compute_A_B()
C = np.linalg.solve(A, B)

print(f"Найденные коэффициенты: C1 = {C[0]:.6f}, C2 = {C[1]:.6f}")


# Приближенное решение
def u_approx(x):
    return phi0(x) + C[0] * phi1(x) + C[1] * phi2(x)


# Точное решение
def u_exact(x):
    return x / (x + 1)


# Проверка в точках
test_points = np.linspace(0, 1, 10)
print("\nСравнение с точным решением:")
for x in test_points:
    approx = u_approx(x)
    exact = u_exact(x)
    print(f"x = {x:.2f}: Приближенно = {approx:.6f}, Точное = {exact:.6f}, Ошибка = {abs(approx - exact):.6f}")

# Проверка краевых условий
print("\nПроверка краевых условий:")
print(f"u'(0) приближенно: {phi0_prime(0) + C[0] * phi1_prime(0) + C[1] * phi2_prime(0):.6f} (ожидается 1.0)")
print(f"u(1) приближенно: {u_approx(1.0):.6f} (ожидается 0.5)")