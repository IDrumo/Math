import sympy as sp
import numpy as np


class LagrangeFunction:
    def __init__(self):
        # Инициализируем пустую целевую функцию и переменные
        self.optimization_func = 0
        self.variables = []
        # Список ограничений
        self.constraints = []
        # Множители Лагранжа
        self.lagrange_multipliers = []
        # Функция Лагранжа
        self.lagrange_expr = self.optimization_func

    def set_optimization_function(self, func):
        self.optimization_func = func

    def set_optimization_function_with_matrix(self, A, b):
        """Устанавливает целевую функцию f(x) = 1/2 * (x^T * A * x) - b * x."""
        n = A.shape[0]  # Размерность переменных
        self.variables = sp.symbols(f'x0:{n}')  # Создаем переменные x0, x1, ...

        # Создаем выражение для целевой функции
        x = np.array(sp.Matrix(self.variables))
        self.optimization_func = 0.5 * (x.T @ A @ x) - (b.T @ x)

        # Обновляем функцию Лагранжа
        self.update_lagrange()

    def add_constraints(self, constraints):
        # Добавляем ограничения из списка
        for constraint in constraints:
            self.constraints.append(constraint)
            # Создаем новый множитель Лагранжа
            lambda_symbol = sp.symbols(f'lambda_{len(self.constraints) - 1}')
            self.lagrange_multipliers.append(lambda_symbol)
        # Обновляем функцию Лагранжа
        self.update_lagrange()

    def update_lagrange(self):
        # Обновляем выражение функции Лагранжа
        self.lagrange_expr = self.optimization_func
        for i, constraint in enumerate(self.constraints):
            self.lagrange_expr += self.lagrange_multipliers[i] * constraint

    def get_lagrange_expression(self):
        return self.lagrange_expr
