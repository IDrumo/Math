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
    return sp.Add(*[(x[i] - x0[i]) ** 2 for i in range(len(x))]) - r ** 2


def analyze_hessian(lagrange_function, vars, x_star):
    """
    Вычисляет Гессиан функции Лагранжа и определяет тип критической точки.

    Args:
        lagrange_function: Функция Лагранжа, для которой нужно вычислить Гессиан.
        vars: Переменные, по которым вычисляется Гессиан.
        x_star: Найденные значения переменных (включая λ).

    Returns:
        Определитель Гессиана и строка с типом критической точки.
    """

    # Определяем Гессиан
    hessian_func = manual_hessian(lagrange_function, vars)

    # Вычисляем определитель Гессиана
    det_hessian = hessian_func.det().subs({vars[i]: x_star[vars[i]] for i in range(len(vars))})

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
      vars: Переменные, по которым будет производиться оптимизация.
      tol: Точность.
      max_iter: Максимальное количество итераций.

    Returns:
      Решение и количество итераций.
    """

    x = np.array(x0, dtype=float)  # Преобразуем x0 в массив NumPy для удобства

    for i in range(max_iter):
        # Вычисляем Гессиан функции Лагранжа
        hess = manual_hessian(func, vars)

        # Подставляем текущее значение x в Гессиан и вычисляем его обратную матрицу
        hess_eval = np.array(hess.subs({vars[j]: x[j] for j in range(len(vars))}), dtype=float)

        # print(hess_eval)
        hess_inv = np.linalg.inv(hess_eval)

        # Вычисляем градиент функции Лагранжа
        grad = np.array(gradient(func, vars).subs({vars[j]: x[j] for j in range(len(vars))}), dtype=float)

        # Проверяем выполнение условия сходимости
        if np.linalg.norm(grad) < tol:  # Используем норму для проверки
            return x, i

        step = hess_inv @ grad  # Находим W^{-1}(x_k) * f(x_k)
        x = x - step.T[0]  # Обновляем x

    return x, max_iter


if __name__ == "__main__":
    # Ввод данных
    A = np.array([[1, 2, 3, 4],
                  [2, 5, 6, 7],
                  [3, 6, 8, 9],
                  [4, 7, 9, 10]])
    # A = np.array([
    #     [0, 2, 0, 0],
    #     [2, -6, 1, 0],
    #     [0, 1, -6, 0],
    #     [0, 0, 0, -1],
    # ])

    b = np.array([1, 2, 3, 4])

    x0 = np.array([0, 0, 0, 0])

    r = 10

    # Определение переменных
    x1, x2, x3, x4, lambd = sp.symbols('x1 x2 x3 x4 lambd')
    x = [x1, x2, x3, x4]
    vars = [lambd, x1, x2, x3, x4]

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
    solutions = sp.solve(equations[1:], vars[1:])  # Не решаем для λ, так как λ = 0

    # Выводим решения
    print("Решения системы уравнений:" + str(solutions))
    print(solutions)
    solutions[lambd] = 0

    # Анализ Гессиана
    det_hessian, type_point = analyze_hessian(lagrange_function, vars, solutions)

    # Выводим определитель и тип критической точки
    print(f"Определитель Гессиана: {det_hessian}")
    print(type_point)

#----------------------------------lambda /= 0-------------------------------------------
    # Генерация начальных приближений
    initial_guesses = []
    for i in range(4):
        for sign in [1, -1]:  # Используем 10 и -10
            x = np.zeros(5)
            x[i+1] = sign * r  # Устанавливаем значение 10 или -10 для i-й переменной
            initial_guesses.append(x)

    # Вывод начальных приближений
    for idx, guess in enumerate(initial_guesses):
        print(f"Начальное приближение {idx + 1}: {guess}")

    # Применение метода Ньютона к каждому начальному приближению
    for idx, x0 in enumerate(initial_guesses):
        solution, iterations = newton_method_multidimensional(lagrange_function, x0, vars)
        print(f"Решение для начального приближения {idx + 1}: {solution}, Итерации: {iterations}")




