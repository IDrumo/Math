import numpy as np
from scipy.optimize import linprog
from tabulate import tabulate


class SimplexMethod:
    def __init__(self, A, b, c):
        self.A = [row[:] for row in A]
        self.b = b[:]
        self.c = c[:]
        self.m = len(A)
        self.n = len(A[0]) if A else 0
        self._create_initial_tableau()

    def _create_initial_tableau(self):
        self.A_slack = [row + [1 if i == j else 0 for j in range(self.m)] for i, row in enumerate(self.A)]
        self.c_extended = self.c + [0] * self.m
        self.tableau = []
        for i in range(self.m):
            self.tableau.append(self.A_slack[i] + [self.b[i]])
        self.tableau.append([-ci for ci in self.c_extended] + [0])
        self.basis = list(range(self.n, self.n + self.m))

    def _pivot(self, row, col):
        pivot_element = self.tableau[row][col]
        if pivot_element == 0:
            raise ValueError("Нулевой опорный элемент.")
        self.tableau[row] = [element / pivot_element for element in self.tableau[row]]
        for r in range(len(self.tableau)):
            if r != row:
                factor = self.tableau[r][col]
                self.tableau[r] = [
                    self.tableau[r][i] - factor * self.tableau[row][i]
                    for i in range(len(self.tableau[r]))
                ]
        self.basis[row] = col

    def _find_pivot_column(self):
        last_row = self.tableau[-1][:-1]
        min_value = min(last_row)
        if min_value >= -1e-10:
            return None
        return last_row.index(min_value)

    def _find_pivot_row(self, pivot_col):
        ratios = []
        for i in range(self.m):
            element = self.tableau[i][pivot_col]
            if element > 1e-10:
                ratio = self.tableau[i][-1] / element
                ratios.append(ratio)
            else:
                ratios.append(float('inf'))
        min_ratio = min(ratios)
        if min_ratio == float('inf'):
            return None
        return ratios.index(min_ratio)

    def solve(self):
        iteration = 0
        while True:
            iteration += 1
            print(f"\nИтерация {iteration}:")
            self._print_tableau()

            pivot_col = self._find_pivot_column()
            if pivot_col is None:
                print("Оптимальное решение найдено.")
                break

            pivot_row = self._find_pivot_row(pivot_col)
            if pivot_row is None:
                raise ValueError("Задача неограничена.")

            print(f"Поворот: Строка {pivot_row}, Столбец {pivot_col}")
            self._pivot(pivot_row, pivot_col)

        solution = [0] * (self.n + self.m)
        for i in range(self.m):
            basic_var = self.basis[i]
            if basic_var < len(solution):
                solution[basic_var] = self.tableau[i][-1]

        x = solution[:self.n]
        z = self.tableau[-1][-1]
        return x, z

    def get_dual_solution(self):
        # Двойственные переменные соответствуют коэффициентам слэк-переменных в строке целевой функции
        y = self.tableau[-1][self.n:self.n + self.m]
        # Значение целевой функции (в последней ячейке симплекс-таблицы)
        z = self.tableau[-1][-1]
        return y, z

    def _print_tableau(self):
        headers = [f"x{j + 1}" for j in range(self.n + self.m)] + ["b"]
        table = []
        for i, row in enumerate(self.tableau):
            if i < self.m:
                basis_var = f"x{self.basis[i] + 1}"
            else:
                basis_var = " z "
            formatted_row = [f"{val:8.3f}" for val in row]
            table.append([basis_var] + formatted_row)
        print(tabulate(table, headers=["Базис"] + headers, tablefmt="pretty"))

    def check_primal_feasibility(self, x):
        for i in range(self.m):
            lhs = sum(self.A[i][j] * x[j] for j in range(self.n))
            if lhs > self.b[i] + 1e-10:
                print(f"Ограничение {i + 1} не выполняется: {lhs} > {self.b[i]}")
                return False
        return True


class Checker:
    def __init__(self, A, b, c):
        self.A = [row[:] for row in A]
        self.b = b[:]
        self.c = c[:]

        self.c_dual = b
        self.A_dual = [[-A[i][j] for i in range(len(A))] for j in range(len(A[0]))]
        self.b_dual = [-ci for ci in self.c]

    def check_initial(self):
        print("Матрица ограничений (A):")
        for row in self.A:
            print(row)
        print("\nПравая часть ограничений (b):")
        print(self.b)
        print("\nКоэффициенты целевой функции (c):")
        print(self.c)

    def strait_solution(self, simplex):
        # global i, val
        solution, objective_value = simplex.solve()
        print("\nПрямая задача:")
        for i, val in enumerate(solution):
            print(f"x{i + 1} = {val:.6f}")
        print(f"Значение целевой функции: {objective_value:.6f}")
        is_feasible = simplex.check_primal_feasibility(solution)
        if is_feasible:
            print("Найденное решение удовлетворяет всем ограничениям (выполнимо).")
        else:
            print("Найденное решение не удовлетворяет всем ограничениям (невыполнимо).")

    def true_strait_solution(self):
        # linprog решает задачу минимизации функции, так что для поиска максимума, я разворачиваю коэффициенты
        # ограничений и ищу минимум, а потом переворачиваю найденный минимум, таким образом находя максимум
        #
        res_builtin = linprog(-np.array(self.c), A_ub=self.A, b_ub=self.b, bounds=(0, None), method='highs')
        if res_builtin.success:
            print("\nРешение с использованием встроенного метода:")
            for i, val in enumerate(res_builtin.x):
                print(f"x{i + 1} = {val:.6f}")
            print(f"Значение целевой функции: {-res_builtin.fun:.6f}")
        else:
            print("Ошибка при решении встроенным методом:", res_builtin.message)

    def true_dual_solution(self):
        global i, val
        res_dual = linprog(self.c_dual, A_ub=self.A_dual, b_ub=self.b_dual, bounds=(0, None), method='highs')
        print("\nДвойственная задача (с использованием scipy.optimize.linprog):")
        if res_dual.success:
            print(f"Оптимальное значение: {res_dual.fun:.6f}")
            print("Оптимальные переменные y:")
            for i, val in enumerate(res_dual.x):
                print(f"y{i + 1} = {val:.6f}")
        else:
            print("Ошибка:", res_dual.message)

    def dual_solution(self, simplex):
        # global i, val
        dual_solution, dual_objective_value = simplex.get_dual_solution()
        print("\nОптимальное двойственное решение (из симплекс-метода):")
        for i, val in enumerate(dual_solution):
            print(f"y{i + 1} = {val:.6f}")
        print(f"Значение целевой функции (двойственной задачи): {dual_objective_value:.6f}")


def big_m_method(c, A, b, M=1000):
    """
    Решение двойственной задачи линейного программирования с помощью М-метода.

    Параметры:
    c (list): Вектор коэффициентов при переменных.
    A (list): Матрица коэффициентов при ограничениях.
    b (list): Вектор правых частей ограничений.
    M (int): Коэффициент при искусственных переменных (по умолчанию 1000).

    Возвращает:
    x (list): Оптимальное решение.
    """
    # Добавление искусственных переменных
    n = len(c)
    m = len(b)
    A_art = np.hstack((np.eye(m), np.zeros((m, n))))
    c_art = [-M] * m
    A = np.hstack((A, A_art))
    c = c + c_art

    # Решение задачи с помощью симплекс-метода
    x = simplex_method(c, A, b)

    # Удаление искусственных переменных
    x = x[:n]

    return x


def simplex_method(c, A, b):
    """
    Решение задачи линейного программирования с помощью симплекс-метода.

    Параметры:
    c (list): Вектор коэффициентов при переменных.
    A (list): Матрица коэффициентов при ограничениях.
    b (list): Вектор правых частей ограничений.

    Возвращает:
    x (list): Оптимальное решение.
    """
    n = len(c)
    m = len(b)
    x = [0] * n
    B = np.eye(m)
    N = np.arange(n)

    while True:
        # Вычисление базисной матрицы
        B_inv = np.linalg.inv(B)

        # Вычисление цен базисных переменных
        c_B = np.dot(c[:m], B_inv)

        # Вычисление цен небазисных переменных
        c_N = np.array(c[m:]) - np.dot(np.array(c[:m]), B_inv) @ np.array(A[:m, m:])[:, :len(c[m:])]

        # Выбор входящей переменной
        j = np.argmax(c_N) + m

        # Вычисление коэффициентов при входящей переменной
        alpha = np.dot(B_inv, A[:, j])

        # Вычисление коэффициентов при базисных переменных
        beta = np.dot(B_inv, b)

        # Проверка оптимальности
        if all(c_N <= 0):
            break

        # Выбор выходящей переменной
        i = np.argmin(beta / alpha)

        # Обновление базисной матрицы
        B[:, i] = A[:, j]
        B_inv = np.linalg.inv(B)

        # Обновление цен базисных переменных
        c_B = np.dot(c[:m], B_inv)

        # Обновление цен небазисных переменных
        c_N = np.array(c[m:]) - np.dot(np.array(c[:m]), B_inv) @ np.array(A[:m, m:])

        # Обновление решения
        x[j] = beta[i] / alpha[i]
        x[N] -= alpha * x[j]

    return x


if __name__ == '__main__':
    A = [[2, 4],
         [1, 1],
         [2, 1]]
    b = [560, 170, 300]
    c = [4, 5]

    # A = [[2, 1, 2],
    #      [4, 1, 1]]
    # b = [-4, -5]
    # c = [560, 170, 300]

    x = big_m_method(c, A, b)
    print("Оптимальное решение:", x)

    checker = Checker(A, b, c)
    checker.check_initial()

    simplex = SimplexMethod(A, b, c)

    checker.strait_solution(simplex)
    # Проверка с использованием встроенного метода
    checker.true_strait_solution()

    checker.dual_solution(simplex)
    checker.true_dual_solution()
