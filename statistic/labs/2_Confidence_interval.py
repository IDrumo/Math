import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as dist


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), dist.sem(a)
    h = se * dist.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def main():
    lamb = 1
    n = 30
    rv_expon = dist.expon(lamb)
    x = np.arange(0.1, 5, 0.1)
    sample = rv_expon.rvs(n)

    # Вычисление доверительного интервала
    alpha = 0.05  # Допустимый уровень ошибки

    mean, interval_l, interval_r = mean_confidence_interval(sample)

    # Отображение результата
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    # Построение гистограммы
    ax.hist(sample, bins=10, density=True, alpha=0.5)

    # Построение графика распределения
    ax.plot(x, rv_expon.pdf(x), 'r-', lw=2)

    # Отображение доверительного интервала
    ax.vlines(interval_l, 0, 1, 'g', label='Доверительный интервал')
    ax.vlines(interval_r, 0, 1, 'g')

    # Тест на функционал доверительного интервала
    # Отображение среднего по выборке и матожидания
    ax.vlines(np.mean(sample), 0, 1, 'blue', label='среднее арифметическое')
    ax.vlines(rv_expon.mean(), 0, 1, 'red', label='матожидание')

    # Оформление графика
    ax.set_xlabel('Значения')
    ax.set_ylabel('Плотность вероятности')
    ax.set_title('Гистограмма и распределение')
    ax.grid(True)
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
