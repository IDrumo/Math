from math import sqrt

import numpy as np

from calcus_math.pract.support.monotonous_running import tridiagonal_solver


class SplineTuple:
    def __init__(self, a, b, c, d, x):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x = x


# Построение сплайна
# x - узлы сетки, должны быть упорядочены по возрастанию, кратные узлы запрещены
# y - значения функции в узлах сетки
# n - количество узлов сетки
def BuildSpline(x, y, n):
    # Инициализация массива сплайнов
    splines = [SplineTuple(0, 0, 0, 0, 0) for _ in range(0, n)]
    for i in range(0, n):
        splines[i].x = x[i]
        splines[i].a = y[i]

    splines[0].c = splines[n - 1].c = 0.0

    # Решение СЛАУ относительно коэффициентов сплайнов c[i] методом прогонки для трехдиагональных матриц
    # Вычисление прогоночных коэффициентов - прямой ход метода прогонки
    alpha = [0.0 for _ in range(0, n - 1)]
    beta = [0.0 for _ in range(0, n - 1)]

    for i in range(1, n - 1):
        hi = x[i] - x[i - 1]
        hi1 = x[i + 1] - x[i]
        A = hi
        C = 2.0 * (hi + hi1)
        B = hi1
        F = 6.0 * ((y[i + 1] - y[i]) / hi1 - (y[i] - y[i - 1]) / hi)
        z = (A * alpha[i - 1] + C)
        alpha[i] = -B / z
        beta[i] = (F - A * beta[i - 1]) / z

    # Нахождение решения - обратный ход метода прогонки
    for i in range(n - 2, 0, -1):
        splines[i].c = alpha[i] * splines[i + 1].c + beta[i]

    # По известным коэффициентам c[i] находим значения b[i] и d[i]
    for i in range(n - 1, 0, -1):
        hi = x[i] - x[i - 1]
        splines[i].d = (splines[i].c - splines[i - 1].c) / hi
        splines[i].b = hi * (2.0 * splines[i].c + splines[i - 1].c) / 6.0 + (y[i] - y[i - 1]) / hi
    return splines


# Вычисление значения интерполированной функции в произвольной точке
def Interpolate(splines, x):
    if not splines:
        return None  # Если сплайны ещё не построены - возвращаем NaN

    n = len(splines)
    s = SplineTuple(0, 0, 0, 0, 0)

    if x <= splines[0].x:  # Если x меньше точки сетки x[0] - пользуемся первым эл-тов массива
        s = splines[0]
    elif x >= splines[n - 1].x:  # Если x больше точки сетки x[n - 1] - пользуемся последним эл-том массива
        s = splines[n - 1]
    else:  # Иначе x лежит между граничными точками сетки - производим бинарный поиск нужного эл-та массива
        i = 0
        j = n - 1
        while i + 1 < j:
            k = i + (j - i) // 2
            if x <= splines[k].x:
                j = k
            else:
                i = k
        s = splines[j]

    dx = x - s.x
    # Вычисляем значение сплайна в заданной точке по схеме Горнера (в принципе, "умный" компилятор применил бы схему Горнера сам, но ведь не все так умны, как кажутся)
    return s.a + (s.b + (s.c / 2.0 + s.d * dx / 6.0) * dx) * dx




def cubic_spline(x, y, A, B):
    n = len(x)
    h = np.diff(x)

    print(h)

    # Создаем матрицу системы уравнений
    A_sys = np.zeros((n, n))
    b_sys = np.zeros(n)

    # Первое краевое условие
    A_sys[0, 0] = 2
    A_sys[0, 1] = 1
    b_sys[0] = 6 / h[0] * ((y[1] - y[0]) / h[0] - A)

    # Внутренние уравнения
    for i in range(1, n - 1):
        A_sys[i, i - 1] = h[i - 1] / 6
        A_sys[i, i] = (h[i - 1] + h[i]) / 3
        A_sys[i, i + 1] = h[i] / 6
        b_sys[i] = (y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]
        b_sys[i] *= 6 / (h[i - 1] + h[i])

    # Последнее краевое условие
    A_sys[-1, -2] = h[-1] / 6
    A_sys[-1, -1] = 2 / 3
    b_sys[-1] = 6 / h[-1] * (B - (y[-1] - y[-2]) / h[-1])

    # Решаем систему уравнений
    M = tridiagonal_solver(A_sys, b_sys)

    # Вычисляем коэффициенты сплайна
    c = (y[1:] - y[:-1]) / h - h * (2 * M[:-1] + M[1:]) / 6
    a = M[:-1]
    b = (M[1:] - M[:-1]) / (3 * h)

    return x, y, a, b, c, M


def cubic_interp1d(x0, x, y):
    """
    Выполняет кубическую интерполяцию одномерной функции с использованием сплайнов.
      x0 : float или 1d-array
      x : (N,) array_like
          1-D массив действительных значений.
      y : (N,) array_like
          1-D массив действительных значений. Длина y вдоль
          оси интерполяции должна быть равна длине x.

    Реализует прием для генерации матрицы Cholesky L
    трехдиагональной матрицы A (таким образом, L является
    бидиагональной матрицей, которую можно решить в два отдельных цикла).
    """
    # Преобразование входных данных в массивы NumPy
    x = np.asfarray(x)
    y = np.asfarray(y)

    # Удаление непрерывных значений (не используется в данном коде)
    # indexes = np.isfinite(x)
    # x = x[indexes]
    # y = y[indexes]

    # Проверка, что массив x отсортирован, и сортировка, если необходимо
    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

    size = len(x)

    # Вычисление разностей между последовательными элементами массивов x и y
    xdiff = np.diff(x)
    ydiff = np.diff(y)

    # Выделение буферных матриц для вычисления коэффициентов сплайна
    Li = np.empty(size)
    Li_1 = np.empty(size - 1)
    z = np.empty(size)

    # Заполнение диагоналей Li и Li-1 и решение системы [L][y] = [B]
    Li[0] = sqrt(2 * xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0  # Граничное условие "естественный сплайн"
    z[0] = B0 / Li[0]

    for i in range(1, size - 1, 1):
        Li_1[i] = xdiff[i - 1] / Li[i - 1]
        Li[i] = sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
        Bi = 6 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    i = size - 1
    Li_1[i - 1] = xdiff[-1] / Li[i - 1]
    Li[i] = sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
    Bi = 0.0  # Граничное условие "естественный сплайн"
    z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    # Решение системы [L.T][x] = [y]
    i = size - 1
    z[i] = z[i] / Li[i]
    for i in range(size - 2, -1, -1):
        z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]

    # Поиск индекса x0 в массиве x
    index = x.searchsorted(x0)
    np.clip(index, 1, size - 1, index)

    # Извлечение значений для вычисления кубического сплайна
    xi1, xi0 = x[index], x[index - 1]
    yi1, yi0 = y[index], y[index - 1]
    zi1, zi0 = z[index], z[index - 1]
    hi1 = xi1 - xi0

    # Вычисление кубического сплайна
    f0 = zi0 / (6 * hi1) * (xi1 - x0) ** 3 + \
         zi1 / (6 * hi1) * (x0 - xi0) ** 3 + \
         (yi1 / hi1 - zi1 * hi1 / 6) * (x0 - xi0) + \
         (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x0)
    return f0

def build_spline(x, y, A, B):
    """
    Функция, строящая кубический сплайн через заданные точки (x, y) с использованием первого типа краевого условия.

    Параметры:
    x - массив x-координат узловых точек
    y - массив y-координат узловых точек
    A - значение первой производной в начальной точке
    B - значение первой производной в конечной точке

    Возвращает:
    коэффициенты сплайна: a, b, c, M
    """
    n = len(x) - 1  # количество интервалов
    h = [x[i+1] - x[i] for i in range(n)]  # длины интервалов
    print(h)

    # Построение системы линейных уравнений для моментов
    A_sys = np.zeros((n+1, n+1))
    b_sys = np.zeros(n+1)

    # Краевое условие в начальной точке
    A_sys[0, 0] = 2
    A_sys[0, 1] = 1
    b_sys[0] = 6/h[0] * ((y[1] - y[0])/h[0] - A)

    # Внутренние точки
    for i in range(1, n):
        A_sys[i, i-1] = h[i-1] / 6
        A_sys[i, i] = (h[i-1] + h[i]) / 3
        A_sys[i, i+1] = h[i] / 6
        b_sys[i] = 6/((h[i-1] + h[i])) * ((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])

    # Краевое условие в конечной точке
    A_sys[n, n-1] = h[n-1] / 6
    A_sys[n, n] = 2
    b_sys[n] = 6/h[n-1] * (B - (y[n] - y[n-1])/h[n-1])

    # Решение системы линейных уравнений
    M = np.linalg.solve(A_sys, b_sys)

    # Вычисление коэффициентов сплайна
    a = [M[i] for i in range(n)]
    b = [(M[i+1] - M[i]) / h[i] for i in range(n)]
    c = [(y[i+1] - y[i])/h[i] - h[i]/6 * (2*M[i] + M[i+1]) for i in range(n)]

    return a, b, c, M



def build_spline2(x, y, A, B):
    """
    Построение кубического сплайна через моменты

    Parameters:
    x (array): узлы интерполирования
    y (array): значения функции в узлах
    A (float): значение первой производной в начале интервала
    B (float): значение первой производной в конце интервала

    Returns:
    M (array): моменты сплайна
    """
    N = len(x) - 1
    h = np.diff(x)
    M = []
    lambda_i = h[:-1] / (h[:-1] + h[1:])
    mu_i = h[1:] / (h[:-1] + h[1:])
    F = np.zeros((N + 1, N + 1))
    F[0, 0] = 2
    F[0, 1] = 1
    F[N, N - 1] = 1
    F[N, N] = 2
    for i in range(1, N):
        F[i, i - 1] = lambda_i[i - 1]
        F[i, i] = 2
        F[i, i + 1] = mu_i[i - 1]
    b = np.zeros(N + 1)
    b[0] = 6 / h[0] * ((y[1] - y[0]) / h[0] - A)
    for i in range(1, N):
        b[i] = 6 / (h[i] + h[i + 1 if i < N-1 else N-1]) * ((y[i + 1 if i < N-1 else N-1] - y[i]) / h[i + 1 if i < N-1 else N-1] - (y[i] - y[i - 1]) / h[i])
    M.append(np.linalg.solve(F, b))
    b[-1] = 6 / h[-1] * (y[-1] - y[-2] - h[-1] * M[-1][-1])
    M.append(np.linalg.solve(F, b))
    M = np.array(M)
    return M


def interpolate(x, y, M, x_new):
    """
    Интерполирование функции сплайном

    Parameters:
    x (array): узлы интерполирования
    y (array): значения функции в узлах
    M (array): моменты сплайна
    x_new (array): точки, в которых нужно интерполировать функцию

    Returns:
    y_new (array): интерполированные значения функции
    """
    N = len(x) - 1
    y_new = np.zeros(len(x_new))
    for j in range(N):
        h = x[j + 1] - x[j]
        mask = (x_new >= x[j]) & (x_new < x[j + 1])
        y_new[mask] = y[j] + (x_new[mask] - x[j]) * ((y[j + 1] - y[j]) / h - h / 6 * (2 * M[j] + M[j + 1])) + (x_new[mask] - x[j]) * (h / 6 * (2 * M[j] + M[j + 1]) - M[j]) + h ** 2 / 2 * M[j] + h ** 3 / 6 * (M[j + 1] - M[j])
    return y_new