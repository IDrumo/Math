import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as dist
from statsmodels.distributions.empirical_distribution import ECDF, _conf_set

plt.rcParams['figure.figsize'] = [12, 8]


def my_ecdf(data):
    x = np.sort(data)
    n = data.size
    y = np.arange(1, n + 1) / n
    return x, y


def my_conf_bands(data, alpha):
    x, y = my_ecdf(data)
    n = data.size
    eps = np.sqrt(1 / (2 * n) * np.emath.log(2 / alpha))
    lower_bond = y - eps
    upper_bond = y + eps
    return lower_bond, upper_bond


def my_bootstrap(data, function, alpha):
    # Создаем 10000 бутстрэп-выборок
    n = len(data)
    # Случайно выбираем из выборки новые выборки длинной n с повторениями
    # применяем к полученной выборке функцию и повторяем 10к раз
    boot_stats = [function(np.random.choice(data, size=n, replace=True)) for _ in range(10000)]

    # Вычисляем границы доверительного интервала
    boot_stats.sort()
    lower = np.percentile(boot_stats, alpha / 2 * 100)
    upper = np.percentile(boot_stats, (1 - alpha / 2) * 100)

    return [lower, upper]


def main():
    mu = 165
    sigma = 10
    n1 = 100
    n2 = 1000
    rv_norm = dist.norm(mu, sigma)
    sample_100 = rv_norm.rvs(n1)
    sample_1000 = rv_norm.rvs(n2)

    x, y = my_ecdf(sample_100)

    fig, ax = plt.subplots(1, 2)
    ax.flat[0].plot(x, rv_norm.cdf(x))
    ax.flat[0].plot(x, y)
    ax.flat[0].set_title('Sample n=100')

    x, y = my_ecdf(sample_1000)

    ax.flat[1].plot(x, rv_norm.cdf(x))
    ax.flat[1].plot(x, y)
    ax.flat[1].set_title('Sample n=1000')

    # plt.show()
    plt.close('all')

    # А теперь с непараметрической доверительной полосой

    alpha = 0.95

    fig, ax = plt.subplots(1, 2)

    x, y = my_ecdf(sample_100)

    ax.flat[0].plot(x, rv_norm.cdf(x))
    ax.flat[0].plot(x, y)
    l, u = my_conf_bands(sample_100, alpha)
    ax.flat[0].plot(x, l)
    ax.flat[0].plot(x, u)
    ax.flat[0].set_title('Sample n=100')

    x, y = my_ecdf(sample_1000)

    ax.flat[1].plot(x, rv_norm.cdf(x))
    ax.flat[1].plot(x, y)
    l, u = my_conf_bands(sample_1000, alpha)
    ax.flat[1].plot(x, l)
    ax.flat[1].plot(x, u)
    ax.flat[1].set_title('Sample n=1000')

    # plt.show()
    plt.close('all')

    # Оценим медиану методом Bootstrap

    chosen_sample = sample_1000
    sample = (chosen_sample,)
    bootstrap = dist.bootstrap(sample, np.median, confidence_level=0.95, method='percentile')
    bootstrap_mine = my_bootstrap(chosen_sample, np.median, 0.05)
    # print(bootstrap.confidence_interval)
    # print(bootstrap_mine)
    plt.plot(x, rv_norm.pdf(x), 'blue')

    vline_size = 0.04
    plt.vlines(mu, 0, vline_size, 'blue', label='True Median')
    plt.vlines(np.median(chosen_sample), 0, vline_size, 'Yellow', label='Sample Median')
    plt.vlines([bootstrap.confidence_interval[0], bootstrap.confidence_interval[1]], 0, vline_size, 'red',
               label='Confidence interval')
    # plt.vlines([bootstrap_mine[0], bootstrap_mine[1]], 0, vline_size, 'green',
    #            label='My Confidence interval') # МОЙ ОЧЕНЬ ХОРОШИЙ!!!

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
