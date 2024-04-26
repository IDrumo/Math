import numpy as np

from calcus_math.pract.support.monotonous_running import tridiagonal_solver, checker


def main():

    A = np.array([
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 2]
    ])
    b = np.array([1, 2, 3])

    # A = np.array([
    #     [3, 6, 0],
    #     [1, 12, 2],
    #     [0, 3, 2]
    # ])
    # b = np.array([4, 7, 12])

    x = tridiagonal_solver(A, b)
    print(x)
    print(checker(A, b, x))


if __name__ == "__main__":
    main()
