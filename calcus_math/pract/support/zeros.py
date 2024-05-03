import sympy as sp

def find_roots(f, a, b, epsilon, flag='all'):
    """
    Находит корни функции f(x) на интервале [a, b] с заданной точностью epsilon.

    Параметры:
    f (sympy.Expr): Функция, для которой ищутся корни.
    a (float): Начало интервала поиска.
    b (float): Конец интервала поиска.
    epsilon (float): Требуемая точность приближения корня.
    flag (str, optional): Определяет, какие корни возвращать:
        'all' - все корни (по умолчанию)
        'positive' - только положительные корни
        'negative' - только отрицательные корни
        'min' - наименьший корень
        'max' - наибольший корень

    Возвращает:
    list: Список найденных корней.
    """
    x = sp.Symbol('x')
    roots = []

    # Находим интервалы, содержащие корни
    intervals = find_intervals(f, a, b)

    # Находим корни на каждом интервале
    for interval in intervals:
        root = find_root_on_interval(f, interval, epsilon)
        roots.append(root)

    # Фильтруем корни в соответствии с флагом
    roots = filter_roots(roots, flag)

    return roots

def find_intervals(f, a, b):
    """
    Находит интервалы на отрезке [a, b], содержащие корни функции f(x).

    Параметры:
    f (sympy.Expr): Функция, для которой ищутся интервалы.
    a (float): Начало отрезка поиска.
    b (float): Конец отрезка поиска.

    Возвращает:
    list: Список интервалов, содержащих корни.
    """
    x = sp.Symbol('x')
    intervals = []
    step = (b - a) / 1000
    x_prev = a
    f_prev = f.subs(x, x_prev)
    for x_curr in [a + i*step for i in range(1, 1001)]:
        f_curr = f.subs(x, x_curr)
        if f_prev * f_curr < 0:
            intervals.append((x_prev, x_curr))
        elif abs(f_curr) < 1e-10:
            intervals.append((x_curr, x_curr))
        x_prev = x_curr
        f_prev = f_curr
    return intervals

def find_root_on_interval(f, interval, epsilon):
    """
    Находит корень функции f(x) на заданном интервале с помощью комбинации
    методов касательных и секущих.

    Параметры:
    f (sympy.Expr): Функция, для которой ищется корень.
    interval (tuple): Интервал, на котором ищется корень.
    epsilon (float): Требуемая точность приближения корня.

    Возвращает:
    float: Найденный корень.
    """
    x = sp.Symbol('x')
    x_left, x_right = interval
    while abs(x_right - x_left) > epsilon:
        f_left = f.subs(x, x_left)
        f_right = f.subs(x, x_right)
        if f_left * sp.diff(f, x).subs(x, x_left) >= 0:
            x_new = x_left - f_left / sp.diff(f, x).subs(x, x_left)
            x_right = x_right - f_right * (x_left - x_right) / (f_left - f_right)
            x_left = x_new
        else:
            x_new = x_right - f_right / sp.diff(f, x).subs(x, x_right)
            x_left = x_left - f_left * (x_right - x_left) / (f_right - f_left)
            x_right = x_new
    return (x_left + x_right) / 2

def filter_roots(roots, flag):
    """
    Фильтрует список корней в соответствии с флагом.

    Параметры:
    roots (list): Список найденных корней.
    flag (str): Определяет, какие корни возвращать.

    Возвращает:
    list: Отфильтрованный список корней.
    """
    if flag == 'positive':
        return [root for root in roots if root > 0]
    elif flag == 'negative':
        return [root for root in roots if root < 0]
    elif flag == 'min':
        return [min(roots)]
    elif flag == 'max':
        return [max(roots)]
    else:
        return roots
