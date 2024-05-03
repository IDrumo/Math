from sympy import symbols, exp
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from calcus_math.pract.support.zeros import find_roots


def main():
    # a, b = 1.0, 1.5
    # n = 100
    #
    # x = np.linspace(a, b, n)  # разбить отрезок [a, b] на n отрезков
    # y = (x - 1) ** 2 - np.exp(-x)  # функция
    #
    # plt.figure(figsize=(10, 6))
    #
    # plt.plot(x, y, "black", label='Оригинальный график')
    #
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.grid()
    # plt.show()

    a, b = -5, 5

    x = sp.Symbol('x')
    f = 1.2 * x ** 2 - sp.sin(10*x)

    roots = find_roots(f, a, b, 1e-6)
    print(roots)  # Выведет: [0.6823236728942158, -0.8413123364471079]

    # roots = find_roots(f, -2, 2, 1e-6, 'positive')
    # print(roots)  # Выведет: [0.6823236728942158]
    #
    # roots = find_roots(f, -2, 2, 1e-6, 'negative')
    # print(roots)  # Выведет: [-0.8413123364471079]
    #
    # roots = find_roots(f, -2, 2, 1e-6, 'min')
    # print(roots)  # Выведет: [-0.8413123364471079]
    #
    # roots = find_roots(f, -2, 2, 1e-6, 'max')
    # print(roots)  # Выведет: [0.6823236728942158]


if __name__ == "__main__":
    main()
