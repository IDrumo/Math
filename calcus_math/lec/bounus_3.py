import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Определяем символьную переменную
x = sp.symbols('x')

# Определяем функцию
f = 1 / sp.sin(x) ** 2

# Метод Паде для аппроксимации функции f(x) = 1/sin^2(x)
pade_approx = sp.apart(f, x, full=True).expand()
# Чтобы конвертировать ряд в функцию, используем lambda
pade_func = sp.lambdify(x, pade_approx, 'numpy')

# Вычисляем ряд Тейлора для функции f в точке 0, порядок 5
taylor_series = f.series(x, 0, 6).removeO()  # removeO убирает символ O большое, который представляет остаточный член ряда

# Чтобы конвертировать ряд в функцию, используем lambda
taylor_func = sp.lambdify(x, taylor_series, 'numpy')

x_values = np.linspace(0.1, 20, 40)
taylor_values = np.vectorize(taylor_func)(x_values)
pade_values = np.vectorize(pade_func)(x_values)

plt.plot(x_values, 1/np.sin(x_values)**2, label='1/sin(x)**2 (exact)')
plt.plot(x_values, taylor_values, 'r--', label='Taylor approximation')
plt.plot(x_values, pade_values, 'g:', label='Pade approximation')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
