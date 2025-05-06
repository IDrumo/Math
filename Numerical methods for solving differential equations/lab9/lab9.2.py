import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve


# Базисные функции: 1, x, x²
def phi(i, x):
    return x ** i  # i=0:1, i=1:x, i=2:x²


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


# Вычисление невязки
integral = np.array([np.sum(x * s ** 2 * u_approx(s)) * dx for s in x])
residual = u_approx(x) - (1 + x * integral)
residual_norm = np.sqrt(np.sum(residual ** 2) * dx)

# Вывод результатов
print(f"Коэффициенты: C1={C[0]:.4f}, C2={C[1]:.4f}, C3={C[2]:.4f}")
print(f"Норма невязки: {residual_norm:.4e}")

# Визуализация невязки
plt.figure(figsize=(10, 5))
plt.plot(x, residual, label='Невязка', color='red')
plt.xlabel('x')
plt.ylabel('R(x)')
plt.title('График невязки')
plt.grid(True)
plt.legend()
# plt.show()
