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

    print(corr(sample1, sample2)) # коэффициент корреляции написанный
    print(np.corrcoef(sample1, sample2)[0, 1]) # коэффициент корреляции встроенный
    # print(np.cov(sample1, sample2)[0, 1]) # ковариация встроенная

    x = np.linspace(0, 100, 1000)

    plt.figure(figsize=(10, 6))

    plt.plot(x, rv_norm1.pdf(x), color='red', label='PDF 1')
    plt.plot(x, rv_norm2.pdf(x), color='blue', label='PDF 2')

    bin_nums = 10
    plt.hist(sample1, bins=bin_nums, density=True, alpha=0.5, color='red', label='Гистограмма 1')
    plt.hist(sample2, bins=bin_nums, density=True, alpha=0.5, color='blue', label='Гистограмма 2')

    plt.xlabel('Значения X')
    plt.ylabel('Вероятность Х')
    plt.title('Сравнение распределений')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
