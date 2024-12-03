import numpy as np


def rotation_method(A: np.array, p: int = 10) -> np.array:
    # По умолчанию ставлю p=10 - количество преград.
    # Чем больше преград, тем лучше сходимость и точность метода, но больше вычислений,
    # а следовательно и более высокое время выполнения.
    def sgn(value):
        return 1 if value >= 0 else -1

    def d(i, j):
        return np.sqrt((A[i, i] - A[j, j]) ** 2 + 4 * A[i, j] ** 2)

    def c(i, j):
        return np.sqrt(0.5 * (1 + abs(A[i, i] - A[j, j]) / d(i, j)))

    def s(i, j):
        return sgn(A[i, j] * (A[i, i] - A[j, j])) * np.sqrt(0.5 * (1 - abs(A[i, i] - A[j, j]) / d(i, j)))

    def step(C, i, j):
        n = A.shape[0]
        C.fill(0)
        for k in range(n):
            for l in range(n):
                if k != i and k != j and l != i and l != j:
                    C[k, l] = A[k, l]
                elif k != i and k != j:
                    C[k, i] = c(i, j) * A[k, i] + s(i, j) * A[k, j]
                    C[i, k] = C[k, i]
                    C[k, j] = -s(i, j) * A[k, i] + c(i, j) * A[k, j]
                    C[j, k] = C[k, j]

        C[i, i] = (c(i, j) ** 2) * A[i, i] + \
                  2 * c(i, j) * s(i, j) * A[i, j] + \
                  (s(i, j) ** 2) * A[j, j]

        C[j, j] = (s(i, j) ** 2) * A[i, i] - \
                  2 * c(i, j) * s(i, j) * A[i, j] + \
                  (c(i, j) ** 2) * A[j, j]

        C[i, j] = 0
        C[j, i] = 0

    def find_max():
        max_abs_value = 0
        indexes = [0, 0]
        n = A.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > max_abs_value:
                    max_abs_value = abs(A[i, j])
                    indexes[0], indexes[1] = i, j
        return indexes[0], indexes[1]

    def stop_condition(sigma):
        for i in range(A.shape[0]):
            for j in range(i + 1, A.shape[0]):
                if abs(A[i, j]) > min(sigma):
                    return False
        return True

    def sigmas():
        return [np.sqrt(max(A[i, i] for i in range(A.shape[0]))) / (10 ** k) for k in range(1, p + 1)]

    A = A.copy()
    n = A.shape[0]
    C = np.zeros_like(A)

    sigma = sigmas()
    iterat = 0

    while not stop_condition(sigma):
        i, j = find_max()
        step(C, i, j)
        A = C.copy()
        iterat += 1

    return np.sort(np.diag(A)), iterat


def checkMatrix(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return False

    if not np.allclose(matrix, matrix.T):
        return False

    minors = [np.linalg.det(matrix[:i, :i]) for i in range(1, matrix.shape[0] + 1)]
    return all(minor > 0 for minor in minors)


def richardson_method(A: np.array, b: np.array, x: np.array, eps: float = 1e-10, max_iter=100, n=20):
    if not checkMatrix(A):
        print("Матрица не положительно определена и/или не симметрическая")
        return x, 0

    lambdas, i = rotation_method(A)
    lambda_min, lambda_max = lambdas[0], lambdas[-1]
    tau_0 = 2 / (lambda_min + lambda_max)
    eta = lambda_min / lambda_max
    rho_0 = (1 - eta) / (1 + eta)

    iterations = 0

    for iteration in range(max_iter):
        if np.linalg.norm(A @ x - b) <= eps:
            break
        for k in range(1, n + 1):
            v_k = np.cos((2 * k - 1) * np.pi / (2 * n))
            t_k = tau_0 / (1 + rho_0 * v_k)
            x = (b - A @ x) * t_k + x
        iterations += 1

    return x, iterations * n


def check_solution(A: np.array, b: np.array, x: np.array):
    # Получение истинного решения через встроенный метод
    true_solution = np.linalg.solve(A, b)

    # Проверка, насколько решение A @ x - b близко к нулю
    residual = A @ x - b
    norm_residual = np.linalg.norm(residual)

    print("Истинное решение (np.linalg.solve):", true_solution)
    print("Полученное решение x:", x)
    print("Норма остатка (A @ x - b):", norm_residual)
    print()


def check_eigenvalues(A: np.array):
    # Вычисление собственных значений через метод вращения
    computed_eigenvalues, i = rotation_method(A)

    # Вычисление собственных значений через встроенный метод
    true_eigenvalues = np.linalg.eigvals(A)

    print("Выполнен поиск собственных значений. Потребовалось итераций: ", i)
    print("Истинные собственные значения:", np.sort(true_eigenvalues))
    print("Собственные значения, полученные методом вращения:", computed_eigenvalues)

    # Проверка характеристического уравнения
    dets = [np.linalg.det(A - eigenval * np.eye(A.shape[0])) for eigenval in computed_eigenvalues]
    print("Определители для det(A - I*lambda):", dets)
    print("Все определители близки к нулю (погрешность 10го порядка):", np.allclose(dets, 0, atol=1e-10))
    print()


def main():
    matrices = {
        "Матрица 1": (np.array([[-0.168700, 0.353699, 0.008540, 0.733624],
                                [0.353699, 0.056519, -0.723182, -0.076440],
                                [0.008540, -0.723182, 0.015938, 0.342333],
                                [0.733624, -0.076440, 0.342333, -0.045744]]),
                      np.array([-0.943568, -0.744036, 0.687843, 0.857774])),
        "Матрица 2": (np.array([[2.2, 1, 0.5, 2],
                                [1, 1.3, 2, 1],
                                [0.5, 2, 0.5, 1.6],
                                [2, 1, 1.6, 2]]),
                      np.array([5.652, 1.545, -1.420, 0.2226])),
        "Матрица 3": (np.array([[1.00, 0.42, 0.54, 0.66],
                                [0.42, 1.00, 0.32, 0.44],
                                [0.54, 0.32, 1.00, 0.22],
                                [0.66, 0.44, 0.22, 1.00]]),
                      np.array([2.3227, 0.7967, 0.6383, 0.2423])),
        "Проверяющая матрица": (np.array(
            [[2.00, 1.00],
             [1.00, 2.00]]),
                                np.array([4, 5])),
        "Матрица методички": (np.array(
            [[2, 1, 1],
             [1, 2.5, 1],
             [1, 1, 3]
             ]),

                              np.array([1.185089, 4.555030, 1.759839])),
    }

    for name, (A, expected_eigenvalues) in matrices.items():
        print(f"-------------{name}-------------")

        b = np.array([1 for i in range(expected_eigenvalues.size)])
        # x = np.array([1 for i in range(expected_eigenvalues.size)])
        x = np.zeros_like(b)

        x, iterations = richardson_method(A, b, x)

        print(f"Итерационный процесс завершился за {iterations} итераций с решением: {x} \n")

        # Проверка решения
        check_solution(A, b, x)

        # Проверка собственных значений
        check_eigenvalues(A)


if __name__ == '__main__':
    main()
