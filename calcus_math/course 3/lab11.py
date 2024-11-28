import numpy as np


def rotation_method(A: np.array, p: int = 5) -> np.array:
    def sgn(value):
        return 1 if value >= 0 else -1

    def d(i, j):
        return np.sqrt((A[i, i] - A[j, j]) ** 2 + 4 * A[i, j] ** 2)

    def c(i, j):
        return np.sqrt(0.5 * (1 + abs(A[i, i] - A[j, j]) / d(i, j)))

    def s(i, j):
        return sgn(A[i, j] * (A[i, i] - A[j, j])) * np.sqrt(0.5 * (1 - abs(A[i, i] - A[j, j]) / d(i, j)))

    def iteration(C, i, j):
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

    def is_enough(sigma, i, j):
        return all(abs(A[i, j]) < s for s in sigma)

    def sigmas():
        return [np.sqrt(max(A[i, i] for i in range(A.shape[0]))) / (10 ** p_i) for p_i in range(p + 1)]

    n = A.shape[0]
    C = np.zeros_like(A)
    i, j = find_max()
    sigma = sigmas()

    while not is_enough(sigma, i, j):
        iteration(C, i, j)
        A = C.copy()
        i, j = find_max()

    return np.sort(np.diag(C))


def richardson_method(A: np.array, b: np.array, x: np.array, eps: float = 1e-5):
    lambda_min, lambda_max = rotation_method(A)[0], rotation_method(A)[-1]
    tau_0 = 2 / (lambda_min + lambda_max)
    eta = lambda_min / lambda_max
    rho_0 = (1 - eta) / (1 + eta)

    n = 20
    iterations = 0

    while np.linalg.norm(A @ x - b) > eps:
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
    computed_eigenvalues = rotation_method(A)

    # Вычисление собственных значений через встроенный метод
    true_eigenvalues = np.linalg.eigvals(A)

    print("Истинные собственные значения:", np.sort(true_eigenvalues))
    print("Собственные значения, полученные методом вращения:", computed_eigenvalues)

    # Проверка характеристического уравнения
    dets = [np.linalg.det(A - eigenval * np.eye(A.shape[0])) for eigenval in computed_eigenvalues]
    print("Определители для det(A - I*lambda):", dets)
    print("Все определители близки к нулю:", np.allclose(dets, 0, atol=1e-10))
    print()


def main():
    A = np.array([[1.00, 0.42, 0.54, 0.66],
                  [0.42, 1.00, 0.32, 0.44],
                  [0.54, 0.32, 1.00, 0.22],
                  [0.66, 0.44, 0.22, 1.00]])
    b = np.array([1, 1, 1, 1])
    x = np.zeros_like(b)

    x, iterations = richardson_method(A, b, x)

    print(f"Итерационный процесс завершился за {iterations} итераций с решением: {x} \n")

    # Проверка решения
    check_solution(A, b, x)

    # Проверка собственных значений
    check_eigenvalues(A)


if __name__ == '__main__':
    main()
