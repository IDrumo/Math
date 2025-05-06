import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import quad

# Инициализация символьных переменных
x = sp.Symbol('x')
k = sp.Symbol('k')

# Определение подынтегральной функции
a = 0
b = sp.Rational('9/10')  # 0.9 как дробь для точности

cos_term = x * sp.cos(sp.pi * x / 1.8)
sin_arg = x * (5 * sp.pi + 10 * sp.pi * k) / 9
integrand = (cos_term ** 2) * sp.sin(sin_arg)
integrand = x**2 * sp.sin(x * (5*sp.pi + 10*sp.pi*k) / 9)

# Попытка аналитического интегрирования
try:
    integral = sp.integrate(integrand, (x, a, b))
    integral_simplified = sp.simplify(integral)

    # Вывод формулы интеграла
    print("Аналитическое выражение интеграла:")
    sp.pretty_print(integral_simplified)
    print("\nLaTeX представление:")
    print(sp.latex(integral_simplified))

except Exception as e:
    print("Аналитическое решение не найдено. Используем численное интегрирование.")
    # Вывод формулы интеграла
    print("\nФормула интеграла:")
    sp.pretty_print(sp.Integral(integrand, (x, a, b)))


# Численное интегрирование и график
# def num_integrand(x, k_val):
#     """Численная реализация подынтегральной функции"""
#     return x * np.cos(np.pi * x / 1.8) ** 2 * np.sin(x * (5 * np.pi + 10 * np.pi * k_val) / 9)
#
#
# # Пределы интегрирования для численного метода
# a_num = 0.0
# b_num = 0.9
#
# # Построение графика зависимости от k
# k_values = np.linspace(0, 5, 50)
#
# integral_results = []
#
# for k_val in k_values:
#     res, _ = quad(num_integrand, a_num, b_num, args=(k_val,))
# integral_results.append(res)
#
# plt.figure(figsize=(10, 5))
# plt.plot(k_values, integral_results, 'b-')
# plt.title(f"Зависимость интеграла от k (численное интегрирование на [{a_num}, {b_num}])")
# plt.xlabel("k")
# plt.ylabel("Значение интеграла")
# plt.grid(True)
# plt.show()
