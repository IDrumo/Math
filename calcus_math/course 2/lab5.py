import numpy as np

from calcus_math.pract.support.monotonous_running import tridiagonal_solver, checker


def TDMA(a, b, c, f):
    a, b, c, f = tuple(map(lambda k_list: list(map(float, k_list)), (a, b, c, f)))

    alpha = [-b[0] / c[0]]
    beta = [f[0] / c[0]]
    n = len(f)
    x = [0] * n

    for i in range(1, n):
        alpha.append(-b[i] / (a[i] * alpha[i - 1] + c[i]))
        beta.append((f[i] - a[i] * beta[i - 1]) / (a[i] * alpha[i - 1] + c[i]))

    x[n - 1] = beta[n - 1]

    for i in range(n - 1, -1, -1):
        x[i - 1] = alpha[i - 1] * x[i] + beta[i - 1]

    return x


def main():
    # A = np.array([
    #     [2, -1, 0],
    #     [-1, 2, -1],
    #     [0, -1, 2]
    # ])
    # b = np.array([1, 2, 3])

    # A = np.array([
    #     [3, 6, 0],
    #     [1, 12, 2],
    #     [0, 3, 2]
    # ])
    # b = np.array([4, 7, 12])

    # A = np.array([
    #     [1, 2, 0],
    #     [2, 1, 3],
    #     [4, 1, 2],
    #     [0, 2, 1]
    # ])
    # b = np.array([4, 2, 3, 5])

    # A = np.array([
    #     [1, 2, 0, 0],
    #     [2, 1, 3, 0],
    #     [0, 1, 2, 3],
    #     [0, 0, 1, 4]
    # ])
    # b = np.array([4, 2, 3, 5])

    # A = np.array([
    #     [5, -1, 0, 0, 0, 0, 0, 0],
    #     [0, -1, -1, 0, 0, 0, 0, 0],
    #     [0, 9, 2, -7, 0, 0, 0, 0],
    #     [0, 0, 5, 22, -10, 0, 0, 0],
    #     [0, 0, 0, 3, 4, 1, 0, 0],
    #     [0, 0, 0, 0, 10, 26, -9, 0],
    #     [0, 0, 0, 0, 0, -10, 18, 1],
    #     [0, 0, 0, 0, 0, 0, 4, 4],
    # ])
    # b = np.array([4, -2, 4, 17, 8, 27, 9, 8])

    # A = np.array([
    #     [4, 4, 0, 0, 0, 0],
    #     [6, 1, -5, 0, 0, 0],
    #     [0, -8, 0, 8, 0, 0],
    #     [0, 0, -8, 12, -4, 0],
    #     [0, 0, 0, -9, 1, 10],
    #     [0, 0, 0, 0, 13/3, -5/3],
    # ])
    # b = np.array([8, 2, 0, 0, 2, 8/3])

    A = np.array([
        [2, 2, 0, 0, 0, 0, 0, 0, 0],
        [-5, 1, 4, 0, 0, 0, 0, 0, 0],
        [0, 3, 7, -10, 0, 0, 0, 0, 0],
        [0, 0, -2, 2, 4, 0, 0, 0, 0],
        [0, 0, 0, -4, 10, -6, 0, 0, 0],
        [0, 0, 0, 0, 7, 0, 7, 0, 0],
        [0, 0, 0, 0, 0, -9, 12, -3, 0],
        [0, 0, 0, 0, 0, 0, -4, 5, 9],
        [0, 0, 0, 0, 0, 0, -8, 0, 19],
    ])
    b = np.array([4, 0, 0, 4, 0, 0, 0, 10, 11])

    x = tridiagonal_solver(A, b)
    print(x)
    print(checker(A, b, x))
    print(x.sum())


def dev():
    # Входные данные
    a = [1.0, 2.0, 3.0, 4.0]
    b = [-2.0, -3.0, -4.0, -5.0]
    c = [1.0, 2.0, 3.0, 4.0]
    f = [1.0, 2.0, 3.0, 4.0]

    # Вызов функции TDMA
    x = TDMA(a, b, c, f)

    # Вывод результата
    print(x)


if __name__ == "__main__":
    main()
    # dev()
