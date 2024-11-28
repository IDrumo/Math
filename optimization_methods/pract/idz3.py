import numpy as np
import sympy as sp


def generate_lagrange_function(f, g, lambd):
    lagrange_function = f + lambd * g
    return lagrange_function


# Функция для вычисления Гессиана
def manual_hessian(func, variables):
    n = len(variables)
    hessian = sp.Matrix(n, n, lambda i, j: sp.diff(sp.diff(func, variables[i]), variables[j]))
    return hessian


def g(x, x0, r):
    return sp.Add(*[sp.sqrt(x[i] - x0[i]) for i in range(len(x))]) - r


def analyze_hessian(lagrange_function, vars, solution):
    """
    Вычисляет Гессиан функции Лагранжа и определяет тип критической точки.

    Args:
        lagrange_function: Функция Лагранжа, для которой нужно вычислить Гессиан.
        vars: Переменные, по которым вычисляется Гессиан.
        solution: Найденные значения переменных (включая λ).

    Returns:
        Определитель Гессиана и строка с типом критической точки.
    """

    # Определяем Гессиан
    hessian_func = manual_hessian(lagrange_function, vars)

    # Подставляем найденные значения в функцию Лагранжа, при этом λ = 0
    substituted_lagrange_function = lagrange_function.subs({vars[-1]: 0})  # vars[-1] — это λ

    # Определяем Гессиан
    hessian_func = manual_hessian(substituted_lagrange_function, vars)

    # Вычисляем определитель Гессиана
    det_hessian = hessian_func.det().subs({vars[i]: solution[vars[i]] for i in range(len(vars) - 1)})

    # Проверяем знак определителя и определяем тип критической точки
    if det_hessian > 0:
        type_point = "Точка является локальным максимумом."
    elif det_hessian < 0:
        type_point = "Точка является локальным минимумом."
    else:
        type_point = "Точка неопределена."

    return det_hessian, type_point


def gradient(func, vars):
    return sp.Matrix([sp.diff(func, var) for var in vars])


def f(x, A, b):
    """
    Функция оптимизации
    """
    return 0.5 * x.T @ A @ x + b.T @ x


def newton_method_multidimensional(func, x0, vars, tol=1e-6, max_iter=100):
    """
    Метод Ньютона для многомерного случая поиска нулевого значения функции Лагранжа.

    Args:
      func: Функция оптимизации.
      x0: Начальная точка (вектор).
      tol: Точность.
      max_iter: Максимальное количество итераций.

    Returns:
      Решение и количество итераций.
    """

    x = np.array(x0, dtype=float)  # Преобразуем x0 в массив NumPy для удобства

    for i in range(max_iter):
        # Вычисляем Гессиан функции Лагранжа
        hess = manual_hessian(func, vars)

        sp.pprint(hess)

        # Подставляем текущее значение x в Гессиан
        hess_eval = hess.subs({vars[i]: x[i] for i in range(len(vars))}).inv()

        # Вычисляем градиент функции Лагранжа
        grad = gradient(lagrange_function, vars).subs({vars[j]: x[j] for j in range(len(vars))})

        # Проверяем выполнение условия сходимости
        if np.linalg.norm(grad) < tol:
            return x, i

        # Метод Гаусса: решаем систему уравнений для W^{-1}(x_k) * f(x_k)
        step = np.linalg.solve(hess_eval, grad)  # Находим W^{-1}(x_k) * f(x_k)
        x = x - step  # Обновляем x

    return x, max_iter


if __name__ == "__main__":
    # Ввод данных
    A = np.array([[1, 2, 3, 4],
                  [2, 5, 6, 7],
                  [3, 6, 8, 9],
                  [4, 7, 9, 10]])

    b = np.array([1, 2, 3, 4])

    x0 = np.array([0, 0, 0, 0])

    r = 1.0

    # Определение переменных
    x1, x2, x3, x4, lambd = sp.symbols('x1 x2 x3 x4 lambd')
    x = [x1, x2, x3, x4]
    vars = [x1, x2, x3, x4, lambd]

    # Определение функции и ограничений
    func = f(np.array([x1, x2, x3, x4]), A, b)  # Используем функцию f с A и b
    g = g(x, x0, r)  # функция ограничения

    # Генерация функции Лагранжа
    lagrange_function = generate_lagrange_function(func, g, lambd)

    gradient_lagrange_function = gradient(lagrange_function, vars)

#----------------------------------lambda = 0-------------------------------------------
    # Условие: приравниваем градиент к нулю и подставляем λ = 0
    gradient_lagrange_function_zero_lambda = gradient_lagrange_function.subs(lambd, 0)

    # Создаем систему уравнений
    equations = [sp.Eq(grad, 0) for grad in gradient_lagrange_function_zero_lambda]

    # Решаем систему уравнений
    solutions = sp.solve(equations[:-1], vars[:-1])  # Не решаем для λ, так как λ = 0

    # Выводим решения
    print("Решения системы уравнений:" + str(solutions))

    # Анализ Гессиана
    det_hessian, type_point = analyze_hessian(lagrange_function, vars, solutions)

    # Выводим определитель и тип критической точки
    print(f"Определитель Гессиана: {det_hessian}")
    print(type_point)

#----------------------------------lambda /= 0-------------------------------------------
    x0 = np.array([0, 0, 0, 0, 0])

    solution, iterations = newton_method_multidimensional(lagrange_function, x0, vars)

    print(f"Решение: {solution}, Итерации: {iterations}")




