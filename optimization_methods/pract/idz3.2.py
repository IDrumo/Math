import numpy as np
import sympy as sp


def generate_lagrange_function(f, g, lambd):
    return f + lambd * g


def manual_hessian(func, variables):
    n = len(variables)
    hessian = sp.Matrix(n, n, lambda i, j: sp.diff(sp.diff(func, variables[i]), variables[j]))
    return hessian


def g(x, x0, r):
    return sp.Add(*[(x[i] - x0[i]) ** 2 for i in range(len(x))]) - r ** 2


def analyze_hessian(lagrange_function, vars, x_star):
    """
    Вычисляет Гессиан функции Лагранжа и определяет тип критической точки
    на основе главных миноров.

    Args:
        lagrange_function: Функция Лагранжа, для которой нужно вычислить Гессиан.
        vars: Переменные, по которым вычисляется Гессиан.
        x_star: Найденные значения переменных (включая λ).

    Returns:
        Список главных миноров и строка с типом критической точки.
    """
    # Определяем Гессиан
    hessian_func = manual_hessian(lagrange_function, vars)

    # Подставляем значения в Гессиан
    hessian_eval = hessian_func.subs({vars[i]: x_star[vars[i]] for i in range(len(vars))})

    return analyze_matrix(hessian_eval)


def analyze_matrix(matrix):
    matrix = sp.Matrix(matrix)
    minors = main_mirror(matrix)
    type_point = matrix_type_for_minors(minors)
    return minors, type_point


def matrix_type_for_minors(minors):
    # Определяем тип критической точки по знакам главных миноров
    if all(minor > 0 for minor in minors):  # Все главные миноры положительные
        type_point = "Точка является локальным минимумом."
    elif all(minors[i] * (-1) ** i > 0 for i in range(len(minors))):  # Чередование знаков
        type_point = "Точка является локальным максимумом."
    else:
        type_point = "Точка неопределена."
    return type_point


def main_mirror(matrix):
    # Вычисляем главные миноры
    minors = []
    for i in range(1, matrix.shape[0] + 1):
        minor = matrix[:i, :i].det()
        minors.append(minor)
    return minors


def gradient(func, vars):
    return sp.Matrix([sp.diff(func, var) for var in vars])


def f(x, A, b):
    return 0.5 * x.T @ A @ x + b.T @ x


def newton_method_multidimensional(func, x0, vars, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)

    for i in range(max_iter):
        hess = manual_hessian(func, vars)
        hess_eval = np.array(hess.subs({vars[j]: x[j] for j in range(len(vars))}), dtype=float)
        hess_inv = np.linalg.inv(hess_eval)

        grad = np.array(gradient(func, vars).subs({vars[j]: x[j] for j in range(len(vars))}), dtype=float)

        if np.linalg.norm(grad) < tol:
            return x, i

        step = hess_inv @ grad
        x = x - step.T[0]

    return x, max_iter


if __name__ == "__main__":
    A = np.array([[1, 2, 3, 4],
                  [2, 5, 6, 7],
                  [3, 6, 8, 9],
                  [4, 7, 9, 10]], dtype=float)

    # print(analyze_matrix(A))

    b = np.array([1, 2, 3, 4])
    x0 = np.array([0, 0, 0, 0])
    r = 10

    x1, x2, x3, x4, lambd = sp.symbols('x1 x2 x3 x4 lambd')
    x = [x1, x2, x3, x4]
    vars = [lambd] + x  # λ идет первым

    func = f(np.array([x1, x2, x3, x4]), A, b)
    constraint_function = g(x, x0, r)

    lagrange_function = generate_lagrange_function(func, constraint_function, lambd)
    gradient_lagrange_function = gradient(lagrange_function, vars)

    # Условие: приравниваем градиент к нулю и подставляем λ = 0
    gradient_lagrange_function_zero_lambda = gradient_lagrange_function.subs(lambd, 0)
    equations = [sp.Eq(grad, 0) for grad in gradient_lagrange_function_zero_lambda]

    solutions = sp.solve(equations[1:], vars[1:])  # Не решаем для λ, так как λ = 0
    print("Решения системы уравнений:" + str(solutions))
    solutions[lambd] = 0

    minors, type_point = analyze_hessian(lagrange_function, vars, solutions)
    print(f"Главные миноры Гессиана: {minors}")
    print(type_point)

    # Генерация начальных приближений
    initial_guesses = []
    for i in range(4):
        for sign in [1, -1]:
            x = np.zeros(5)
            x[i + 1] = sign * r
            initial_guesses.append(x)

    for idx, guess in enumerate(initial_guesses):
        print(f"Начальное приближение {idx +  1}: {guess}")

        solution, iterations = newton_method_multidimensional(lagrange_function, guess, vars)
        print(f"Решение для начального приближения {idx + 1}: {solution}, Итерации: {iterations}")

        # Отображаем значение функции в найденном приближении
        function_value = lagrange_function.subs({vars[j]: solution[j] for j in range(len(vars))})
        print(f"Функция в этой точке: {function_value}")
        print()