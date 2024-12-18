from scipy.optimize import linprog
from tabulate import tabulate


class SimplexMethod:
    def __init__(self, A, b, c):
        """
        Инициализация симплекс-метода.

        :param A: Матрица ограничений (A) размера (m x n).
        :param b: Вектор правых частей ограничений (b) размера (m).
        :param c: Вектор коэффициентов целевой функции (c) размера (n).
        """
        self.A = [row[:] for row in A]  # Копирование матрицы ограничений
        self.b = b[:]
        self.c = c[:]
        self.m = len(A)  # Количество ограничений
        self.n = len(A[0]) if A else 0  # Количество переменных
        self._create_initial_tableau()

    def _create_initial_tableau(self):
        """
        Создает начальную симплекс-таблицу, добавляя слэк-переменные.
        """
        # Добавляем слэк-переменные (единичная матрица)
        self.A_slack = [row + [1 if i == j else 0 for j in range(self.m)] for i, row in enumerate(self.A)]

        # Расширяем вектор c для слэк-переменных
        self.c_extended = self.c + [0] * self.m

        # Создаём симплекс-таблицу
        # Таблица имеет m+1 строк и n+m+1 столбец
        self.tableau = []
        for i in range(self.m):
            self.tableau.append(self.A_slack[i] + [self.b[i]])
        # Добавляем строку целевой функции
        self.tableau.append([-ci for ci in self.c_extended] + [0])

        # Инициализируем базисные переменные как слэк-переменные
        self.basis = list(range(self.n, self.n + self.m))

    def _pivot(self, row, col):
        """
        Выполняет операцию поворота (pivot) в таблице симплекса.

        :param row: Номер строки (индекс) для поворота.
        :param col: Номер столбца (индекс) для поворота.
        """
        pivot_element = self.tableau[row][col]
        if pivot_element == 0:
            raise ValueError("Нулевой опорный элемент.")

        # Нормализуем опорную строку
        self.tableau[row] = [element / pivot_element for element in self.tableau[row]]

        # Обнуляем остальные элементы в столбце
        for r in range(len(self.tableau)):
            if r != row:
                factor = self.tableau[r][col]
                self.tableau[r] = [
                    self.tableau[r][i] - factor * self.tableau[row][i]
                    for i in range(len(self.tableau[r]))
                ]

        # Обновляем базисные переменные
        self.basis[row] = col

    def _find_pivot_column(self):
        """
        Находит индекс столбца для входа в базис (самый отрицательный элемент в строке целевой функции).

        :return: Индекс столбца или None, если оптимальное решение найдено.
        """
        last_row = self.tableau[-1][:-1]
        min_value = min(last_row)
        if min_value >= -1e-10:
            return None
        return last_row.index(min_value)

    def _find_pivot_row(self, pivot_col):
        """
        Находит индекс строки для выхода из базиса (метод минимального отношения).

        :param pivot_col: Индекс столбца для входа в базис.
        :return: Индекс строки или None, если задача неограничена.
        """
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
        """
        Выполняет симплекс-метод для нахождения оптимального решения.

        :return: Кортеж (решение, значение целевой функции).
        """
        iteration = 0
        while True:
            iteration += 1
            print(f"\nИтерация {iteration}:")
            self._print_tableau()

            pivot_col = self._find_pivot_column()
            if pivot_col is None:
                print("Оптимальное решение найдено.")
                break  # Оптимальное решение найдено

            pivot_row = self._find_pivot_row(pivot_col)
            if pivot_row is None:
                raise ValueError("Задача неограничена.")

            print(f"Поворот: Строка {pivot_row}, Столбец {pivot_col}")
            self._pivot(pivot_row, pivot_col)

        # Извлекаем решение
        solution = [0] * (self.n + self.m)
        for i in range(self.m):
            basic_var = self.basis[i]
            if basic_var < len(solution):
                solution[basic_var] = self.tableau[i][-1]

        # Только исходные переменные
        x = solution[:self.n]
        # Значение целевой функции
        z = self.tableau[-1][-1]
        return x, z

    def get_dual_solution(self):
        """
        Извлекает решение двойственной задачи из симплекс-таблицы.

        :return: Вектор двойственных переменных y.
        """
        # Двойственные переменные соответствуют коэффициентам слэк-переменных в строке целевой функции
        y = self.tableau[-1][self.n:self.n + self.m]
        return y

    def _print_tableau(self):
        """
        Выводит текущую симплекс-таблицу с использованием библиотеки tabulate.
        """
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
        """
        Проверяет, удовлетворяет ли найденное решение условиям ограничений.

        :param x: Вектор решения.
        :return: True, если решение выполнимо, иначе False.
        """
        for i in range(self.m):
            lhs = sum(self.A[i][j] * x[j] for j in range(self.n))
            if lhs > self.b[i] + 1e-10:
                print(f"Ограничение {i + 1} не выполняется: {lhs} > {self.b[i]}")
                return False
        return True


if __name__ == '__main__':
    # Задаем данные для задачи
    A = [[2, 4],
         [1, 1],
         [2, 1]]  # Ограничения
    b = [560, 170, 300]  # Правая часть ограничений
    c = [4, 5]  # Целевая функция

    print("Матрица ограничений (A):")
    for row in A:
        print(row)
    print("\nПравая часть ограничений (b):")
    print(b)
    print("\nКоэффициенты целевой функции (c):")
    print(c)

    # Решение прямой задачи с помощью симплекс-метода
    simplex = SimplexMethod(A, b, c)
    try:
        solution, objective_value = simplex.solve()
        print("\nПрямая задача:")
        for i, val in enumerate(solution):
            print(f"x{i + 1} = {val:.6f}")
        print(f"Значение целевой функции: {objective_value:.6f}")

        # Проверка на выполнимость (прямую проверку условий ограничений)
        is_feasible = simplex.check_primal_feasibility(solution)
        if is_feasible:
            print("Найденное решение удовлетворяет всем ограничениям (выполнимо).")
        else:
            print("Найденное решение не удовлетворяет всем ограничениям (невыполнимо).")

    except ValueError as e:
        print("Ошибка:", e)

    # Решение двойственной задачи с помощью scipy.optimize.linprog
    # Двойственная задача: минимизация b^T y, при условиях A^T y >= c, y >=0
    # Для linprog: минимизируем c_dual^T y при A_ub y <= b_dual
    # То есть, A_dual = -A^T, b_dual = -c
    c_dual = b
    A_dual = [[-A[i][j] for i in range(len(A))] for j in range(len(A[0]))]
    b_dual = [-ci for ci in c]

    res_dual = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, bounds=(0, None), method='highs')

    print("\nДвойственная задача (с использованием scipy.optimize.linprog):")
    if res_dual.success:
        print(f"Оптимальное значение: {res_dual.fun:.6f}")
        print("Оптимальные переменные y:")
        for i, val in enumerate(res_dual.x):
            print(f"y{i + 1} = {val:.6f}")
    else:
        print("Ошибка:", res_dual.message)

    # Дополнительный вывод двойственного решения из симплекс-метода
    dual_solution = simplex.get_dual_solution()
    print("\nОптимальное двойственное решение (из симплекс-метода):")
    for i, val in enumerate(dual_solution):
        print(f"y{i + 1} = {val:.6f}")
