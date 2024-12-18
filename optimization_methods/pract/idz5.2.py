import numpy as np
from scipy.optimize import linprog


class SimplexMethod:
    def __init__(self, matrix: np.array, b: np.array, c: np.array):
        self._matrix = matrix
        self._b = b
        self._c = c
        self._n, self._m = matrix.shape

    def _make_simplex_table(self):
        self._simplex_table = np.hstack((self._matrix, np.eye(self._n), self._b.reshape(-1, 1)))
        c_extended = np.hstack((self._c, np.zeros(self._n + 1)))
        self._simplex_table = np.vstack((self._simplex_table, -c_extended))

    def _make_dual_simplex_table(self):
        self._simplex_table = np.hstack(  # совмещаем транспонированную исходную матрицу и дополнительный набор векторов
            (
                -self._matrix.T,  # исходная транспонированная матрица
                np.eye(self._m),  # Введение нового базиса
                -self._c.reshape(-1, 1)  # добавление уловий #FIXME почему минус
            )
        )
        c_extended = - np.hstack((self._b, np.zeros(self._m + 1)))
        # for i in range(self._m):
        #     c_extended[-i -2] = 10 ** 3
        self._simplex_table = np.vstack((self._simplex_table, c_extended))

    def _pivot(self, row, col):
        self._simplex_table[row] /= self._simplex_table[row, col]
        for i in range(self._simplex_table.shape[0]):
            if i != row:
                self._simplex_table[i] -= self._simplex_table[i, col] * self._simplex_table[row]

    def compute(self):
        self._make_simplex_table()

        # FIXME На паре было не так. Мы искали максимум по модулю, но и не меняли знак у коэффициентов (см. стр. 15)
        while np.any(self._simplex_table[-1, :-1] < 0):
            # FIXME почему условие на минимум? (должно быть пока любое число в нижней строке больше нуля. Алгоритм завершается, когда все коэфф меньше нуля)
            col = np.argmin(self._simplex_table[-1, :-1])  # FIXME почему не максимум по модулю?

            if np.all(self._simplex_table[:-1, col] <= 0):  # FIXME не разбираюсь
                raise ValueError("Решение неограничено.")

            ratios = self._simplex_table[:-1, -1] / self._simplex_table[:-1, col]
            ratios[self._simplex_table[:-1, col] <= 0] = np.inf  # FIXME не разбираюсь
            row = np.argmin(ratios)
            print(f"Опорный элемент: {self._simplex_table[row, col]}")
            self._pivot(row, col)

        solution = np.zeros(self._m)
        for i in range(self._n):
            basic_col = np.where(self._simplex_table[i, :self._m] == 1)[0]
            if len(basic_col) == 1:
                solution[basic_col[0]] = self._simplex_table[i, -1]

        objective_value = self._simplex_table[-1, -1]
        return solution, objective_value

    # FIXME по сути это и есть тот же самый обход матрицы симплекс метода ПОЛНОСТЬЮ ТОЖЕ САМОЕ
    def compute_dual(self):
        self._make_dual_simplex_table()

        while np.min(self._simplex_table[:-1, -1]) < 0:  # FIXME почему так
            simplex_diff = np.min(self._simplex_table[:-1, -1])
            index_of_element = np.where(self._simplex_table[:, -1] == simplex_diff)[0][0]

            min_element = np.inf
            min_column = 0
            for column in range(self._simplex_table.shape[1] - 1):
                if self._simplex_table[-1, column] == 0:
                    continue
                if self._simplex_table[index_of_element, column] < 0 and abs(
                        self._simplex_table[-1, column] / self._simplex_table[index_of_element, column]) < min_element:
                    min_column = column
                    min_element = abs(self._simplex_table[-1, column] / self._simplex_table[index_of_element, column])
            self._simplex_table[index_of_element, :] /= self._simplex_table[index_of_element, min_column]
            # self._pivot(index_of_element, min_column)
            for line in range(self._simplex_table.shape[0]):
                if line == index_of_element:
                    continue
                self._simplex_table[line, :] -= self._simplex_table[index_of_element, :] * self._simplex_table[
                    line, min_column]
        solution = np.zeros(self._n)
        colms = []
        for i in range(self._n):
            basic_col = np.where(self._simplex_table[:-1, i] == 1)[0]

            if len(basic_col) == 1 and not basic_col[0] in colms:
                colms.append(basic_col[0])
                solution[i] = self._simplex_table[basic_col[0], -1]

        objective_value = self._simplex_table[-1, -1]
        return solution, objective_value


if __name__ == '__main__':
    np.set_printoptions(linewidth=200, suppress=True)

    # np.random.seed(42)
    A = np.random.randint(1, 10, size=(8, 6))
    b = np.random.randint(10, 20, size=8)
    c = np.random.randint(1, 10, size=6)  # Правая часть ограничений

    print("Матрица ограничений (A):")
    print(A)
    print("\nПравая часть ограничений (b):")
    print(b)
    print("\nКоэффициенты целевой функции (c):")
    print(c)

    A = np.array([[2, 4],
                  [1, 1],
                  [2, 1]])
    b = np.array([560, 170, 300])
    c = np.array([4, 5])

    sm = SimplexMethod(A, b, c)
    try:
        solution, objective_value = sm.compute()
        print("\nПрямая задача:")
        print("Решение:", solution)
        print("Значение целевой функции:", objective_value)
    except ValueError as e:
        print("Ошибка:", e)

    try:
        solution, objective_value = sm.compute_dual()
        print("\nДвойственная задача:")
        print("Решение:\n", solution)
        print("Значение целевой функции:", objective_value)
    except ValueError as e:
        print("Ошибка:", e)

    c_dual = b  # целевая функция двойственной задачи
    A_dual = -np.array(A).T  # транспонированные коэффициенты ограничений с отрицательным знаком
    b_dual = -np.array(c)  # коэффициенты двойственных ограничений

    res_dual = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, bounds=(0, None), method='highs')
    print("\nРешение задачи линейного программирования встроенным методом:")
    print("Оптимальное значение:", res_dual.fun)
    print("Оптимальные переменные:\n", res_dual.x)
