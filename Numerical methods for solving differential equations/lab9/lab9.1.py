import numpy as np
from numpy.linalg import solve, norm


# Базисные функции: 1, x, x²
def phi(i, x):
    return x ** i


# Параметры интегрирования
a, b = 0.0, 1.0
N = 1000  # Количество точек
x = np.linspace(a, b, N)
dx = (b - a) / (N - 1)

# Матрица системы и вектор правой части
A = np.zeros((3, 3))
b_vec = np.zeros(3)

# Заполнение матрицы A и вектора b
for i in range(3):
    for j in range(3):
        # Интеграл от phi_i * phi_j
        A[i, j] = np.sum(phi(i, x) * phi(j, x)) * dx
    # Интеграл от x * phi_i(x) dx
    J_i = np.sum(x * phi(i, x)) * dx
    # Вычитаем компоненту, связанную с интегралом ядра
    A[i, :] -= np.array([J_i / 3, J_i / 4, J_i / 5])
    # Правая часть: интеграл от phi_i(x) dx
    b_vec[i] = np.sum(phi(i, x)) * dx

# Решение системы
C = solve(A, b_vec)


# Формирование решения
def u_approx(x):
    return C[0] * phi(0, x) + C[1] * phi(1, x) + C[2] * phi(2, x)


def residual(x, u_approx):
    # Правая часть уравнения
    f = x

    # Левая часть интегрального уравнения
    integral_term = np.zeros_like(x)
    for i in range(3):
        # Интеграл от ядра K(x, t) = t * phi_i(t) умноженного на u_approx(t)
        integral_term += np.sum(phi(i, x) * x * u_approx(phi(i, x)) * dx)
    left_side = u_approx(x) + integral_term

    # Невязка - разность между левой и правой частями
    res = left_side - f
    return res


# Вывод коэффициентов и проверка
print(f"Коэффициенты: C1={C[0]:.4f}, C2={C[1]:.4f}, C3={C[2]:.4f}")
print("Проверка в точках:")
for x_point in [0, 0.5, 1]:
    print(f"x={x_point}: u(x)={u_approx(x_point):.6f}")


residuals = residual(x, u_approx)
print(f"\nМаксимальная невязка: {np.max(np.abs(residuals)):.6e}")
print(f"Средняя невязка: {np.mean(np.abs(residuals)):.6e}")
print(f"L2-норма невязки: {norm(residuals):.6e}")
print(residuals)
