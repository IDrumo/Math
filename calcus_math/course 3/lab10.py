import numpy as np


def is_positive_definite(A: np.array) -> bool:
    if A.shape[0] != A.shape[1]:
        return False

    if not np.allclose(A, A.T):
        return False

    minors = [np.linalg.det(A[:i, :i]) for i in range(1, A.shape[0] + 1)]
    return all(minor > 0 for minor in minors)


def sgn(value):
    return 1 if value >= 0 else -1


def d(A, i, j):
    return np.sqrt((A[i, i] - A[j, j]) ** 2 + 4 * A[i, j] ** 2)


def c(A, i, j):
    return np.sqrt(0.5 * (1 + abs(A[i, i] - A[j, j]) / d(A, i, j)))


def s(A, i, j):
    return sgn(A[i, j] * (A[i, i] - A[j, j])) * np.sqrt(0.5 * (1 - abs(A[i, i] - A[j, j]) / d(A, i, j)))


def iteration(A, C, i, j):
    n = A.shape[0]
    C.fill(0)  # Явная инициализация на каждом шаге
    for k in range(n):
        for l in range(n):
            if k != i and k != j and l != i and l != j:
                C[k, l] = A[k, l]
            elif k != i and k != j:
                C[k, i] = c(A, i, j) * A[k, i] + s(A, i, j) * A[k, j]
                C[i, k] = C[k, i]
                C[k, j] = -s(A, i, j) * A[k, i] + c(A, i, j) * A[k, j]
                C[j, k] = C[k, j]

    C[i, i] = (c(A, i, j) ** 2) * A[i, i] + \
              2 * c(A, i, j) * s(A, i, j) * A[i, j] + \
              (s(A, i, j) ** 2) * A[j, j]

    C[j, j] = (s(A, i, j) ** 2) * A[i, i] - \
              2 * c(A, i, j) * s(A, i, j) * A[i, j] + \
              (c(A, i, j) ** 2) * A[j, j]

    C[i, j] = 0
    C[j, i] = 0


def sigmas(A, p):
    return [np.sqrt(max(A[i, i] for i in range(A.shape[0]))) / (10 ** p_i) for p_i in range(p + 1)]


def find_ij(A):
    max_abs_value = 0
    indexes = [0, 0]
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j]) > max_abs_value:
                max_abs_value = abs(A[i, j])
                indexes[0], indexes[1] = i, j
    return indexes[0], indexes[1]


def is_enough(A, sigma, i, j):
    return all(abs(A[i, j]) < s for s in sigma)


def calculate(A: np.array, p: int) -> np.array:
    n = A.shape[0]
    C = np.zeros_like(A)
    i, j = find_ij(A)
    sigma = sigmas(A, p)

    while not is_enough(A, sigma, i, j):
        iteration(A, C, i, j)
        A = C.copy()
        i, j = find_ij(A)

    return C


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
                      np.array([2.3227, 0.7967, 0.6383, 0.2423]))
    }

    for name, (A, expected_eigenvalues) in matrices.items():
        print(f"Вычисление для {name}:")
        solution = calculate(A, 5)
        computed_eigenvalues = np.linalg.eigvals(solution)
        print("Вычисленные собственные значения:", computed_eigenvalues)
        print("Ожидаемые собственные значения:", expected_eigenvalues)
        print("Собственные значения совпадают:",
              np.allclose(np.sort(computed_eigenvalues), np.sort(expected_eigenvalues), atol=1e-3))
        # Проверка на принадлежность к уравнению det(A - I*lambda) = 0
        dets = [np.linalg.det(A - eigenval * np.eye(A.shape[0])) for eigenval in computed_eigenvalues]
        print("Определители для det(A - I*lambda):", dets)
        print("Все определители близки к нулю:", np.allclose(dets, 0, atol=1e-10))
        print()


if __name__ == '__main__':
    np.set_printoptions(linewidth=200, suppress=True)
    main()
