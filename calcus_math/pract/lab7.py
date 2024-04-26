from sympy import symbols, exp


def main():
    # Определение переменной и функции
    x = symbols('x')
    f = (x - 1) ** 2 - exp(-x)
    a, b = 1.0, 1.5
    n = 100
    h = (b - a) / n

    all_x = [a + i * h for i in range(n + 1)]
    all_y = [f.subs(x, xi) for xi in all_x]


if __name__ == "__main__":
    main()
