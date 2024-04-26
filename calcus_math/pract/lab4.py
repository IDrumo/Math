from sympy import symbols, exp

from calcus_math.pract.support.checker import check_integral
from calcus_math.pract.support.integrations import left_rectangle_rule, gaussian_quadrature, Newton_Cotes_method


def main():
    # Определение переменной и функции
    x = symbols('x')
    f = (x - 1) ** 2 - exp(-x)
    a, b = 1.0, 1.5
    n = 100
    h = (b - a) / n

    all_x = [a + i * h for i in range(n + 1)]
    all_y = [f.subs(x, xi) for xi in all_x]

    # Проверка функций на точность:

    check_integral(a, b, f, x,
                   left_rectangle_rule(a, b, n, f, x),
                   Newton_Cotes_method(a, b, f, x, 6 * 1),
                   gaussian_quadrature(a, b, f, 4 * 1, x))


if __name__ == "__main__":
    main()
