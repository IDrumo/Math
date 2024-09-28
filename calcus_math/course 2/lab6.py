import numpy as np

from calcus_math.pract.support.Splines import cubic_interp1d, build_spline, interpolate, cubic_spline, BuildSpline, \
    Interpolate
import matplotlib.pyplot as plt

if __name__ == '__main__':
    a, b, n = -4, 10, 7
    # a, b, n = 1.0, 1.5, 7

    x = np.linspace(a, b, n) # разбить отрезок [a, b] на n отрезков
    y = (x - 1) ** 2 - np.exp(-x) # функция

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Исходные данные')

    x_new = np.linspace(a, b, n*20) # разбить отрезок [a, b] на 20n отрезков
    y_new = (x_new - 1) ** 2 - np.exp(-x_new)

    spline = BuildSpline(x, y, len(x))

    x_plot = np.linspace(min(x), max(x), 100)
    y_plot = [Interpolate(spline, xi) for xi in x_plot]

    plt.plot(x_plot, y_plot, "red", label='Сплайн')
    plt.plot(x_new, y_new, "black", label='Оригинальный график')



    # Вычисляем краевые значения
    A = np.gradient(y, x)[0]
    B = np.gradient(y, x)[-1]

    x_spline, y_spline, a, b, c, M = cubic_spline(x, y, A, B)
    x_plot = np.linspace(x[0], x[-1], len(x) * 10, endpoint=False)
    y_plot = []
    for i in range(len(x) - 1):
        mask = (x_plot >= x[i]) & (x_plot < x[i + 1])
        y_plot.append(y[i] + c[i] * (x_plot[mask] - x[i]) + b[i] * (x_plot[mask] - x[i]) ** 2 / 2 + a[i] * (
                    x_plot[mask] - x[i]) ** 3 / 6)
    y_plot = np.concatenate(y_plot)
    plt.plot(x_plot, y_plot, "green", label='Отстойный график')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()