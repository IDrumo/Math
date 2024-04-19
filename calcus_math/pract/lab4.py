from support.subs import newton_cotes_coefficient, get_all_NC_coefficient

print(get_all_NC_coefficient(1))


def main():
    # Определение переменной и функции
    x = symbols('x')
    f = (x - 1) ** 2 - exp(-x)
    a, b = 1.0, 1.5
    n = 10
    h = (b - a) / n

if __name__ == "__main__":
    main()