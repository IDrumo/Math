from calcus_math.pract.support.subs import get_all_NC_coefficient


def left_rectangle_rule(a, b, n, function, symbol):
    width = (b - a) / n
    integral = 0
    xi = a
    for i in range(n):
        integral += function.subs(symbol, xi) * width
        xi += width
    return integral


def Newton_Cotes_method(a, b, f, x, n):
    ns_coefs = get_all_NC_coefficient(n)
    lenght = b - a
    step = lenght / n
    sum = 0
    xi = a
    for i in range(n + 1):
        sum += f.subs(x, xi) * ns_coefs[i]
        xi += step
    return lenght * sum


def gaussian_quadrature(a, b, func, n, x):
    """
    Вычислить определенный интеграл функции 'func' на интервале [a, b]
    с использованием Гауссовой квадратуры с n точками.

    Параметры:
    func (callable): функция, которую нужно интегрировать.
    a (float): начало интервала интегрирования.
    b (float): конец интервала интегрирования.
    n (int): количество точек в Гауссовой квадратуре.

    Возвращает:
    float: результат интегрирования.
    """
    # Словарь с корнями и весами для различных n
    Gauss_coefs = {
        1: ([0], [2]),
        2: ([-0.5773502691896257, 0.5773502691896257], [1, 1]),
        3: ([-0.7745966692414834, 0, 0.7745966692414834], [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
        4: ([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
            [0.3478548451374538, 0.6521451548625462, 0.6521451548625462, 0.3478548451374538])
    }

    if n not in Gauss_coefs:
        raise ValueError("Поддерживаются только n=1, 2, 3, 4")

    roots, weights = Gauss_coefs[n]
    result = 0.0

    # Вычисляем интеграл с помощью преобразования координат
    for xi, wi in zip(roots, weights):
        # Преобразовываем корень xi из интервала [-1, 1] в [a, b]
        x_mapped = 0.5 * (b - a) * xi + 0.5 * (a + b)
        result += wi * func.subs(x, x_mapped) * (b - a) / 2

    return result
