import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as dist


def corr(X, Y):
    x_mean = X.sum() / len(X)
    y_mean = Y.sum() / len(Y)
    xy_mean = (X * Y).sum() / len(X)
    sigma_x = np.sqrt(((X - x_mean) ** 2).sum() / len(X))
    sigma_y = np.sqrt(((Y - y_mean) ** 2).sum() / len(Y))

    covariation = (xy_mean - x_mean * y_mean) / (sigma_x * sigma_y)
    return covariation


def main():
    n = 50
    mu1 = 50
    mu2 = 60
    sigma1 = 5
    sigma2 = 10
    rv_norm1 = dist.norm(mu1, sigma1)
    rv_norm2 = dist.norm(mu2, sigma2)
    sample1 = rv_norm1.rvs(n)
    sample2 = rv_norm2.rvs(n)

    print(f'Var(X) = {np.var(sample1)}')
    print(f'Var(Y) = {np.var(sample2)}')
    print(f'Cov(X,Y) = {np.cov(sample1, sample2)[0, 1]}')
    print()
    print(np.cov(sample1, sample2))
    print()
    print(f'Var(X+Y) = {np.var(sample1 + sample2, ddof=1)}')
    print(f'Var(X+Y) = {np.var(sample1, ddof=1) + np.var(sample2, ddof=1) + 2 * np.cov(sample1, sample2, ddof=1)[0, 1]}')
    print(f'Var(X+Y) = {np.cov(sample1, sample2, ddof=1).sum()}')


if __name__ == '__main__':
    main()
