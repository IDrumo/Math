from sympy import *
from support.interpolations import *
from support.checker import *
import matplotlib.pyplot as plt


def main():
    # Определение переменной и функции
    x = symbols('x')
    f = (x - 1) ** 2 - exp(-x)
    a, b = 1.0, 1.5
    n = 10
    h = (b - a) / n

    # Точки для интерполяции
    x_star = 1.07
    x_double_star = 1.02
    x_triple_star = 1.47
    x_quadruple_star = 1.27

    # Заданные точки для интерполяции
    all_x = [a + i * h for i in range(n + 1)]
    all_y = [f.subs(x, xi) for xi in all_x]

    new_f = lagrange(all_x, all_y, x)
    check_results(f, new_f, x, a, b, n, x_triple_star)


if __name__ == "__main__":
    main()
