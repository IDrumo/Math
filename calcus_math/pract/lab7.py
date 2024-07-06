from sympy import symbols, exp
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from calcus_math.pract.support.zeros import find_roots


def main():

    a, b = -5, 5

    x = sp.Symbol('x')
    f = 1.2 * x ** 2 - sp.sin(10*x)

    roots = find_roots(f, a, b, 1e-6)
    print(roots)

    # roots = find_roots(f, -2, 2, 1e-6, 'positive')
    # print(roots)
    #
    # roots = find_roots(f, -2, 2, 1e-6, 'negative')
    # print(roots)
    #
    # roots = find_roots(f, -2, 2, 1e-6, 'min')
    # print(roots)
    #
    # roots = find_roots(f, -2, 2, 1e-6, 'max')
    # print(roots)


if __name__ == "__main__":
    main()
