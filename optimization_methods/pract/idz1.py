import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 5 * x ** 2 - 8 * x ** (5 / 4) - 20 * x


def dichotomy_min_v2(f, a, b, eps):
    count_dichotomy = 0
    x1 = a
    x2 = b
    while abs(x2 - x1) > eps:
        x3 = (x1 + x2) / 2
        count_dichotomy += 2
        if f(x3) < f(x1):
            x1 = x3
        else:
            x2 = x3
    return (x2 - x1) / 2, count_dichotomy


def dichotomy_min(f, a, b, eps):
    count_dichotomy = 0
    # count_dichotomy = np.log((b - a) / eps) / np.log(2)

    delta = eps / 2

    while abs(b - a) > eps:
        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2
        count_dichotomy += 2
        if f(x1) > f(x2):
            a = x1
        else:
            b = x2
    return (b + a) / 2, count_dichotomy


def dichotomy_roots(a, b, eps):  # отрезок от a до b делим на n частей, погрешность eps
    """
    Функция отделения и уточнения корня
    """

    n = 1000
    assert a != 0, 'a равно 0'
    assert b != 0, 'b равно 0'
    # сначала отделим корни
    setka = np.linspace(a, b, n)
    # далее уточним корни
    for x, y in zip(setka, setka[1:]):
        if f(x) * f(y) > 0:  # если на отрезке нет корня, смотрим следующий
            continue
        root = None
        while (abs(f(y) - f(x))) > eps:  # пока отрезок больше заданной погрешности, выполняем нижестоящие операции:
            mid = (y + x) / 2  # получаем середину отрезка
            if f(mid) == 0 or f(mid) < eps:  # если функция в середине отрезка равна нулю или меньше погрешности:
                root = mid  # корень равен серединному значению
                break
            elif (f(mid) * f(x)) < 0:  # Иначе, если произведение функции в середине отрезка на функцию в т. а <0
                y = mid  # серединой становится точка b
            else:
                x = mid  # в другом случае - точка а
        if root:
            yield root


def bine_formula(n):
    return (np.power((1 + np.sqrt(5)) / 2, n) - np.power((1 - np.sqrt(5)) / 2, n)) / np.sqrt(5)


def fibonacci_min(f, a, b, eps):
    n = 0
    while (b - a) / eps > bine_formula(n):
        n += 1

    count_fibonacci = n

    x1 = a + bine_formula(n) / bine_formula(n + 2) * (b - a)
    x2 = a + bine_formula(n + 1) / bine_formula(n + 2) * (b - a)

    f1 = f(x1)
    f2 = f(x2)

    for i in range(2, n):
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (bine_formula(n - i + 1) / bine_formula(n - i + 2)) * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (bine_formula(n - i + 2) / bine_formula(n - i + 2)) * (b - a)
            f2 = f(x2)

    return (a + b) / 2, count_fibonacci


def fibonacci_min_v2(func, a, b, eps):
    n = 0
    while bine_formula(n) < (b-a) / eps:
        n += 1

    x1 = a + (bine_formula(n-2) / bine_formula(n)) * (b-a)
    x2 = a + (bine_formula(n-1) / bine_formula(n)) * (b-a)

    f1 = func(x1)
    f2 = func(x2)

    for i in range(2, n):
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (bine_formula(n-i) / bine_formula(n-i+2)) * (b-a)
            f1 = func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (bine_formula(n-i+1) / bine_formula(n-i+2)) * (b-a)
            f2 = func(x2)

    return (a+b) / 2, n


def plot_dependence(eps_values, dichotomy_counts, fibonacci_counts):
    plt.plot(np.log(eps_values), dichotomy_counts, label='Дихотомия')
    plt.plot(np.log(eps_values), fibonacci_counts, label='Фибоначчи')
    plt.xlabel('Логарифм заданной точности')
    plt.ylabel('Количество вычислений целевой функции')
    plt.legend()
    plt.show()


def plot_function(func, x_range, num_points=1000):
    """
    Отрисовывает график переданной функции на заданном промежутке.

    :param func: Функция, которую нужно отрисовать.
    :param x_range: Кортеж (x_min, x_max) - диапазон по оси X.
    :param num_points: Количество точек для построения графика.
    """
    x_min, x_max = x_range
    x_values = np.linspace(x_min, x_max, num_points)  # Генерация значений по оси X
    y_values = func(x_values)  # Вычисление значений по оси Y

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label=f'y = {func.__name__}(x)')
    plt.title('График функции')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    a, b = 3, 3.5
    eps = 0.001

    # plot_function(f, (0, 6))

    x_dichotomy, _ = dichotomy_min(f, a, b, eps)
    print(f'Минимум Дихотомии: f({x_dichotomy}) = {f(x_dichotomy)}')

    x_fibonacci, _ = fibonacci_min_v2(f, a, b, eps)
    print(f'Минимум Фибоначчи: f({x_fibonacci}) = {f(x_fibonacci)}')

    eps_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.0001]
    dichotomy_counts = []
    fibonacci_counts = []

    for eps in eps_values:
        _, counter = dichotomy_min(f, a, b, eps)
        dichotomy_counts.append(counter)

        _, counter = fibonacci_min_v2(f, a, b, eps)
        fibonacci_counts.append(counter)

    plot_dependence(eps_values, dichotomy_counts, fibonacci_counts)
