import numpy as np


def f(x, y):
    return 2 * y - 2


def exact_solution(x):
    return 1 - np.exp(2*x)


# Метод неявного Рунге-Кутта второго порядка
def runge_kutta_implicit(h, y0, x0, x_end):
    n = int((x_end - x0) / h)
    x_values = np.linspace(x0, x_end, n + 1)
    y_values = np.zeros(n + 1)
    y_values[0] = y0

    for i in range(n):
        x_n = x_values[i]
        u_n = y_values[i]

        # Явный метод Эйлера
        u_euler = u_n + h * f(x_n, u_n)

        # Неявный метод Рунге-Кутта
        k1 = f(x_n, u_n)
        k2 = f(x_n + h, u_euler)
        y_values[i + 1] = u_n + (h / 2) * (k1 + k2)

    return x_values, y_values


# Параметры
x0 = 0
x_end = 1
y0 = exact_solution(x0)

# Шаги
h_values = [0.1, 0.05]

# Сравнение результатов
for h in h_values:
    x_values, y_values = runge_kutta_implicit(h, y0, x0, x_end)
    exact_values = exact_solution(x_values)

    # Погрешность
    errors = np.abs(exact_values - y_values)

    # Вывод результатов
    print(f"Шаг h = {h}:")
    print(f"{'x':<10} {'y (приближенное)':<20} {'y (точное)':<20} {'Ошибка':<20}")
    print("-" * 70)  # Разделительная линия
    for x, y, exact, error in zip(x_values, y_values, exact_values, errors):
        print(f"{x:<10.2f} {y:<20.6f} {exact:<20.6f} {error:<20.6f}")
    print()
