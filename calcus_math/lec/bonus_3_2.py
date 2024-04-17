import sympy as sp
import math
import numpy as np
from scipy.interpolate import pade, lagrange
import matplotlib.pyplot as plt


def paint(start_x, finish_x, point_number, f1, f1_label, f2, f2_label):
    x_vals = np.linspace(start_x, finish_x, point_number)
    plt.plot(x_vals, f1(x_vals), label=f1_label)
    plt.plot(x_vals, f2(x_vals), 'r--', label=f2_label)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()


# Определяем символьную переменную
x = sp.symbols('x')


# Определяем функцию
def f(x):
    return 1 / np.sin(x) ** 2


f2 = 1 / sp.sin(x) ** 2
taylor_series = f2.series(x, 0,
                          9).removeO()  # removeO убирает символ O большое, который представляет остаточный член ряда
# Чтобы конвертировать ряд в функцию, используем lambda
taylor_func = sp.lambdify(x, taylor_series, 'numpy')
# Теперь можно использовать taylor_func как обычную функцию для numpy массивов

paint(0.1, 3, 400, f, 'f(x)', taylor_func, 'Taylor')
paint(0.1, 100, 4000, f, 'f(x)', taylor_func, 'Taylor')

# Коэффициенты ряда для x^2/sin^2(x), начинающегося с 1
coefficients = [1, 0, 1 / 3, 0, 1 / 15, 0, 2 / 189, 0, 1 / 675, 0]
M, N = 4, 4
p, q = pade(coefficients, M)
pade_approx = lambda x: p(x) / q(x) / x ** 2  # обратное преобразование

paint(0.1, 3, 400, f, 'f(x)', pade_approx, 'Pade')
paint(0.1, 100, 4000, f, 'f(x)', pade_approx, 'Pade')

x_list = [0.1 + 0.5 * i for i in range(50)]
y_list = [f(xi) for xi in x_list]
# print(x_list)
# print(y_list)
# lagrange_approx = lagrange(x_list, y_list)
# print(lagrange_approx)
# paint(0.1, 3, 400, f, 'f(x)', lagrange_approx, 'Lagrange')
# paint(0.1, 100, 4000, f, 'f(x)', lagrange_approx, 'Lagrange')
