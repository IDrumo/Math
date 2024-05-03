import numpy as np
from sympy import symbols, simplify
from scipy.optimize import curve_fit


def make_system(xy_table, basis):
    matrix = np.zeros((basis, basis + 1))
    for i in range(basis):
        for j in range(basis):
            sum_a = 0
            sum_b = 0
            for k in range(len(xy_table[0]) // 2):
                sum_a += np.power(xy_table[0, k], i) * np.power(xy_table[0, k], j)
                sum_b += xy_table[1, k] * np.power(xy_table[0, k], i)
            matrix[i, j] = sum_a
            matrix[i, basis] = sum_b
    return matrix


def main():
    # Пример использования
    xy_table = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                         [50, 53, 56, 59, 62, 64, 66, 68, 70, 71, 72, 73, 74]])
    basis = 13
    system_matrix = make_system(xy_table, basis)
    print(system_matrix)

def dev1():
    # Входные данные
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    y = np.array([50, 53, 56, 59, 62, 64, 66, 68, 70, 71, 72, 73, 74])

    # Определение целевой функции
    def g(x, a, b):
        return a + b / (x + 1)

    # Нахождение оптимальных коэффициентов
    popt, pcov = curve_fit(g, x, y)

    # Вывод результатов
    a, b = popt
    print(f"Коэффициенты: a = {a:.2f}, b = {b:.2f}")

    # Вычисление среднеквадратичного отклонения
    y_pred = g(x, a, b)
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"Среднеквадратичное отклонение: {rmse:.2f}")

def dev2():
    import numpy as np

    # Входные данные
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    y = np.array([3000, 3500, 4200, 4800, 5500, 6200, 6800, 7400, 7900, 8300, 8600, 8900, 9200])

    def g(x, a, b):
        return a + b * x

    # Линейная аппроксимация методом наименьших квадратов
    a, b = np.polyfit(x, y, deg=1)

    popt, pcov = curve_fit(g, x, y)

    # Вывод результатов
    a, b = popt

    # Вычисление предсказанных значений
    y_pred = a + b * x

    # Вычисление среднеквадратичного отклонения
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)

    print(f"Коэффициенты: a = {a:.2f}, b = {b:.2f}")
    print(f"Среднеквадратичное отклонение: {rmse:.2f}")


def dev3():
    import numpy as np

    # Входные данные
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    y = np.array([50, 53, 56, 59, 62, 64, 66, 68, 70, 71, 72, 73, 74])

    def g(x, a, b):
        return a + b * x

    # Линейная аппроксимация методом наименьших квадратов
    a, b = np.polyfit(x, y, deg=1)

    popt, pcov = curve_fit(g, x, y)

    # Вывод результатов
    a, b = popt

    # Вычисление предсказанных значений
    y_pred = a + b * x

    # Вычисление среднеквадратичного отклонения
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)

    print(f"Коэффициенты: a = {a:.2f}, b = {b:.2f}")
    print(f"Среднеквадратичное отклонение: {rmse:.2f}")


def dev4():
    import numpy as np

    # Входные данные
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # y = np.array([3000, 3500, 4200, 4800, 5500, 6200, 6800, 7400, 7900, 8300, 8600, 8900, 9200])
    y = np.array([50, 53, 56, 59, 62, 64, 66, 68, 70, 71, 72, 73, 74])

    def g(x, a, b):
        return np.exp(a + b * x)

    popt, pcov = curve_fit(g, x, y)

    # Вывод результатов
    a, b = popt
    print(f"Коэффициенты: a = {a:.2f}, b = {b:.2f}")

    # Вычисление среднеквадратичного отклонения
    y_pred = g(x, a, b)
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"Среднеквадратичное отклонение: {rmse:.2f}")


if __name__ == '__main__':
    # main()
    # dev1()
    # dev2()
    # dev3()
    dev4()


