from calcus_math.pract.support.interpolations import *
from calcus_math.pract.support.subs import *


def check_results(initial_function, interpolated_function, symbol, start_x, finish_x, points_number, x_value):
    """
         Проверить удволетворяет ли полученная интерполяционная функция необходимой погрешности
         : param initial_function: изначальная функция
         : param interpolated_function: интерполированная функция
         : param symbol: переменная в предоставленных формулах
         : param start_x: начало промежутка интерполирования
         : param finish_x: конец промежутка интерполирования
         : param points_number: количество взятых узлов
         : param x_value: Интересующая точка
    """
    x_value += 0
    param = 1

    # Инициализация -------------------------------------------------------------------------------
    step = (finish_x - start_x) / points_number
    x_values = [start_x + i * step for i in range(points_number + 1)]
    y_values = [initial_function.subs(symbol, xi) for xi in x_values]
    x_max_value = x_values[y_values.index(max(y_values))]
    x_min_value = x_values[y_values.index(min(y_values))]
    # ---------------------------------------------------------------------------------------------

    R_min = abs((dnf(initial_function, symbol, points_number - param).subs(symbol, x_min_value)
                 / factorial(points_number - param)
                 * omega(start_x, finish_x, points_number, symbol)).subs(symbol, x_value))

    R_max = abs((dnf(initial_function, symbol, points_number - param).subs(symbol, x_max_value)
                 / factorial(points_number - param)
                 * omega(start_x, finish_x, points_number, symbol, points_number - param)).subs(symbol, x_value))

    R_min, R_max = (R_min, R_max) if R_min <= R_max else (R_max, R_min)
    R = abs(interpolated_function.subs(symbol, x_value) - initial_function.subs(symbol, x_value))

    print(R_min, R, R_max)

    if R_min <= R <= R_max:
        print("Ok")
        return true
    else:
        print("Fail")
        return false


def check_integral(x_start, x_finish, function, symbol, *args):
    """
        Сравнить значение интеграла функции со значениями в массиве
        : param x_start: начало промежутка интегрирования
        : param x_finish: конец промежутка интегрирования
        : param function: исходная функция
        : param symbol: переменная интегрирования
        : param args: массив со сравниваемыми значениями
    """

    func = integrate(function, (symbol, x_start, x_finish)).evalf()

    min_err = abs(func - args[0])
    min_err_index = 0
    max_err = abs(func - args[0])
    max_err_index = 0

    for i, value in enumerate(args):
        err = abs(value - func)

        if err < min_err:
            min_err = err
            min_err_index = i

        if err > max_err:
            max_err = err
            max_err_index = i

        print(f"Погрешность вычисления для значения с индексом {i} равна: {err}")

    print(f"(НАИМЕНЕЕ ТОЧНОЕ) Погрешность: {max_err} у элемента с индексом {max_err_index}")
    print(f"(НАИБОЛЕЕ ТОЧНОЕ) Погрешность: {min_err} у элемента с индексом {min_err_index}")
