import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
import scipy.stats as dist
import math


def my_ecdf(data):
    x = np.sort(data)
    n = data.size
    y = np.arange(1, n + 1) / n
    return x, y


def draw_cdf_graph(distributions, samples=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Определяем общий интервал для всех распределений
    min_val = min([dist.ppf(0.001) for dist in distributions])
    max_val = max([dist.ppf(0.999) for dist in distributions])
    x = np.linspace(min_val, max_val, 1000)

    for dist in distributions:
        y = dist.cdf(x)
        ax.plot(x, y)

    if samples is not None:
        for sample in samples:
            x, y = my_ecdf(sample)
            ax.plot(x, y, linestyle='--')

    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    ax.set_title('Сравнение функций распределения')
    plt.show()


def simple_1side_assimp():
    mu_0 = 11
    mu_A = 11.3
    sigma = 4.3
    n = 30

    normalize_statistic = np.sqrt(n) * (mu_A - mu_0) / sigma  # Нормализованная статистика

    alpha = 0.05
    quantile = dist.norm.ppf(1 - alpha)
    prognoses_value = quantile * sigma / np.sqrt(n) + mu_0

    print(mu_A < prognoses_value)  # True => Сохраняем нулевую гипотезу

    p_value = 1 - dist.norm.cdf(normalize_statistic)  # вероятность получить значение больше наблюдаемого

    print(alpha < p_value)  # True => Сохраняем нулевую гипотезу


def simple_2side_assimp():
    mu_0 = 5
    mu_A = 4.99
    sigma = 0.01
    n = 30

    alpha = 0.05

    quantile1 = dist.norm.ppf(alpha / 2)
    quantile2 = dist.norm.ppf(1 - alpha / 2)
    critical_value_l = quantile1 * sigma / np.sqrt(n) + mu_0
    critical_value_r = quantile2 * sigma / np.sqrt(n) + mu_0

    # print(critical_value_l, mu_A, critical_value_r)
    print(critical_value_l < mu_A < critical_value_r)

    normalize_value = np.sqrt(n) * np.abs(mu_A - mu_0) / sigma
    p_value = (1 - dist.norm.cdf(normalize_value))

    print(alpha / 2, p_value)
    print(alpha / 2 < p_value)


def normal_base_known_general_variation():
    # Создадим 2 выборки из разных норм распр и проверим гипотезу о том, что
    # у второй группы матожидание больше.

    n = 5
    mu_c = 5
    mu_A = 9
    sigma = 2

    rv_c = dist.norm(mu_c, sigma)
    rv_A = dist.norm(mu_A, sigma)

    sample_c = rv_c.rvs(n)
    sample_A = rv_A.rvs(n)

    # ТОЧНАЯ ОЦЕНКА
    # Сумма независимых нормальных случайных величин
    # есть нормальная случайная величина с суммарными матожиданием и дисперсиями
    # т.е. введем новую нормальную случайную величину Z = X - Y и нормализуем её

    # В рамках нулевой гипотезы матожидания равны, а значит их разность равна нулю

    normalize_value = np.sqrt(n) * np.abs(np.mean(sample_c) - np.mean(sample_A) - (0)) / np.sqrt(
        sigma ** 2 + sigma ** 2)

    alpha = 0.05
    p_value = 1 - dist.norm.cdf(normalize_value)

    print(alpha, p_value)
    print(alpha < p_value)  # False => отвергаем гипотезу о равенстве матожиданий


def assimp_normal_base_unknown_general_variation():
    n = 5
    mu_c = 5
    mu_A = 9
    unknown_sigma = 2

    rv_c = dist.norm(mu_c, unknown_sigma)
    rv_A = dist.norm(mu_A, unknown_sigma)

    sample_c = rv_c.rvs(n)
    sample_A = rv_A.rvs(n)

    # АСИМПТОТИЧЕСКИЙ МЕТОД
    # Теорема Слуцкого: нормализованное нормальное распределение с оцененной дисперсией
    # сходится по вероятности к такому же но с истинной дисперсией

    # Тогда аналогично предыдущему
    normalize_value = np.sqrt(n) * np.abs(np.mean(sample_c) - np.mean(sample_A) - (0)) \
                      / np.sqrt(np.var(sample_c, ddof=1) + np.var(sample_A, ddof=1))

    alpha = 0.05
    p_value = 1 - dist.norm.cdf(normalize_value)

    print(alpha, p_value)
    print(alpha < p_value)


def exact_normal_base_unknown_general_variation():
    # Как оценить дисперсию выборки из нормального распределения?
    # По Теореме Кокрана: оценка дисперсии домноженная на размер выборки и деленная на генеральную дисперсию
    # имеет хи-квадрат распределение с n-1 степенями свободы

    n = 5
    mu = 60
    sigma_0 = 3
    sigma_A = 3

    rv_norm = dist.norm(mu, sigma_A)
    sample = rv_norm.rvs(n)

    # Будем делать односторонний тест на равенство диспрерсии 3, с алтернативной гипотезой, что дисперсия больше
    alpha = 0.05
    chi = dist.chi2(n - 1)
    normalize_value = n * np.var(sample, ddof=1) / sigma_0

    p_value = 1 - chi.cdf(normalize_value)
    print(alpha, p_value)
    print(alpha < p_value)


def students_t_test():
    n = 5
    mu_c = 5
    mu_A = 9
    unknown_sigma = 2

    rv_c = dist.norm(mu_c, unknown_sigma)
    rv_A = dist.norm(mu_A, unknown_sigma)

    sample_c = rv_c.rvs(n)
    sample_A = rv_A.rvs(n)

    # ТОЧНЫЙ МЕТОД (не асимптотический)
    # Распределение стьюдента (в пределе сходится к нормальному, но применяется для маленьких выборок)
    # Формулируется как отношение стандартизованной нормальной случайной величины
    # к корню из дисперсии (распределенной как хи-квадрат) деленной на количество случайных величин.

    t_observe = np.sqrt(n) * (sample_c.mean() - sample_A.mean() - (0)) / np.sqrt(
        sample_c.var(ddof=1) + sample_A.var(ddof=1))

    alpha = 0.05
    t_crit = dist.t.ppf(alpha, n)  # односторонний тест

    print(t_observe < t_crit)

    p_value = 1 - dist.t.cdf(t_observe, n)
    print(alpha, p_value)
    print(alpha < p_value)


def normal_base_unknown_different_variation():
    n = 5
    m = 6
    mu_c = 5
    mu_A = 6
    unknown_sigma_c = 2
    unknown_sigma_A = 3

    rv_c = dist.norm(mu_c, unknown_sigma_c)
    rv_A = dist.norm(mu_A, unknown_sigma_A)

    sample_c = rv_c.rvs(n)
    sample_A = rv_A.rvs(m)

    # Это все еще можно привести к t-распределению, (даже когда даны выборки разного размера)
    # Только в качестве степеней свободы выбираем или наименьшее из двух (консервативный подход)
    # Или пользуемся формулой Уэлча-Саттертуэйта

    alpha = 0.05

    df = (sample_c.var(ddof=1) / n + sample_A.var(ddof=1) / m) ** 2 / (
            sample_c.var(ddof=1)**2 / (n ** 2 * (n - 1)) + sample_A.var(ddof=1)**2  / (m ** 2 * (m - 1)))

    t_observe = np.abs(sample_c.mean() - sample_A.mean() - (0)) / np.sqrt(
        sample_c.var(ddof=1) / n + sample_A.var(ddof=1) / m)
    t_crit = dist.t.ppf(1 - alpha / 2, df)

    print(t_observe < t_crit)

    p_value = 1 - dist.t.cdf(t_observe, df)
    print(alpha < p_value)


def verification_of_binomial_asymptotic_requirements(n, p):
    '''
    Функция делает проверку на допустимость использования асимптотических методов
    для биномиально распределенных данных с заданными параметрами

    :param p: вероятность возникновения наблюдаемого события
    :param n: количество наблюдений
    :return:
        True - Данных достаточно для применения асимптотических методов
        False - Нужно больше данных для применения асимптотических методов
    '''
    print(0 < n * p - 3 * np.sqrt(n * p * (1 - p)) < n * p + 3 * np.sqrt(n * p * (1 - p)) < n)


def binomial_assimp_2side_1sample():
    n = 130
    rv_realize = 69
    p_0 = 0.5
    mean = rv_realize / n

    # Необходимое требование для использования асимптотических методов для биномиальных распределений:
    # 0 < n*p - 3 * sqrt(n*p*(1-p)) < n*p + 3 * sqrt(n*p*(1-p) < n

    verification_of_binomial_asymptotic_requirements(n, p_0)

    # просто нормализуем среднее (или просто сумму, это ни на что не влияет)

    normalize_observe_value = np.sqrt(n) * np.abs(mean - p_0) / np.sqrt(p_0 * (1 - p_0))

    alpha = 0.05
    crit_value = dist.norm.ppf(1 - alpha / 2)
    print(normalize_observe_value < crit_value)

    p_value = 1 - dist.norm.cdf(normalize_observe_value)
    print(alpha < p_value)


def binomial_exact_2side_1sample():
    # Метод точный, потому что по данным можно воссаздать функцию распределения вероятности

    p_0 = 0.8
    n = 21
    positive_observe_value = 19
    alpha = 0.05

    verification_of_binomial_asymptotic_requirements(n, p_0)

    rv_binomial = dist.binom(n, p_0)

    crit_value_l = rv_binomial.ppf(alpha / 2)
    crit_value_r = rv_binomial.ppf(1 - alpha / 2)

    x = np.arange(0, n + 1)
    plt.plot(x, rv_binomial.pmf(x))
    plt.vlines([crit_value_l, crit_value_r], 0, 0.2, 'r')
    # plt.show()

    # print(crit_value_l, positive_observe_value, crit_value_r)
    print(crit_value_l < positive_observe_value < crit_value_r)


def binomial_assimp_1side_2sample():
    alpha = 0.05
    n = 49
    m = 51
    sum_x = 28
    sum_y = 38

    # мы можем оценить параметр распределения через MLE.
    # оценка для Биномиального распределения методом MLE будет выглядить как p = (sum_x + sum_y) / (n + m)

    p_mle = (sum_x + sum_y) / (n + m)
    normalize_observe_value = np.sqrt(n * m) * (sum_y / m - sum_x / n) / np.sqrt((n + m) * (p_mle * (1 - p_mle)))
    crit_value = dist.norm.ppf(alpha)

    print(normalize_observe_value < crit_value)

    p_value = 1 - dist.norm.cdf(normalize_observe_value)

    print(alpha < p_value)


def binomial_exact_2side_2sample():
    alpha = 0.05
    n = 49
    m = 51
    sum_x = 28
    sum_y = 38

    # Точный тест Фишера (на базе полиномиального распределения)
    # Преобразуем данные из прошлого теста в табличку:
    #                        Альтернативный вариант     Традиционный вариант    Итого
    # положительное событие             38(a)                    28(b)            66(a+b)
    # отрицательное событие             13(c)                    21(d)            34(c+d)
    # итого                            51(a+c)                  49(b+d)          100(n)
    #
    # тогда вероятность получить данные как в таблице можно посчитать:
    # p = ( (a+b)!(c+d)!(a+c)!(b+d)! ) /( a!b!c!d!n! )

    a = sum_y
    b = sum_x
    c = m - sum_y
    d = n - sum_x

    p_i = []
    for i in range(a, a + c, 1):
        p_i.append(
            (math.factorial(a + b) * math.factorial(c + d) * math.factorial(a + c) * math.factorial(b + d)) / (
                    math.factorial(i) * math.factorial(a + b - i) * math.factorial(a + c - i)
                    * math.factorial(c + d - (a + c - i)) * math.factorial(n + m))
        )

    p_value = sum(p_i)
    print(p_value)

    # Проверка:
    table = np.array([[38, 28], [13, 21]])
    oddsr, p = dist.fisher_exact(table, alternative='greater')
    print(p)

    print(alpha < p_value)


def equals_distributions():
    # Чтобы сравнить между собой 2 распределения восмользуемся тестом Колмогорова-Смирнова
    # D = sup|F_ecdf1 - F_ecdf2|
    # полученная статистика подчиняется распределению Колмогорова-Смирнова

    mu = 5
    sigma = 2
    n = 100

    rv_norm = dist.norm(mu, sigma)
    sample_1 = rv_norm.rvs(n)
    sample_2 = rv_norm.rvs(n)

    # сравниваем то как ecdf похожа на свою cdf

    stat, p_value = dist.kstest(sample_1, rv_norm.cdf, N=n)
    print(stat, p_value)
    draw_cdf_graph([rv_norm], [sample_1])

    # Или как не похожа

    stat, p_value = dist.kstest(sample_1, dist.norm(mu, sigma + 2).cdf, N=n)
    print(stat, p_value)
    draw_cdf_graph([rv_norm, dist.norm(mu, sigma + 2)], [sample_1])

    # Или как похожи 2 ecdf

    stat, p_value = dist.ks_2samp(sample_1, sample_2)
    print(stat, p_value)
    draw_cdf_graph([rv_norm], [sample_1, sample_2])

    # Или можно указать односторонний тест

    stat, p_value = dist.kstest(sample_1, dist.norm(4, 2).cdf, N=n, alternative='less')
    print(stat, p_value)
    draw_cdf_graph([dist.norm(4, 2)], [sample_1])


def dependence_test():
    # H_0: случайные величины независимы.
    # Для проверки гипотезы будем пользоваться Критерием согласия Пирсона
    # D = ((n_i - E_i)**2 / E_i).sum()
    # E_i = N * p_i = N * p_x * p_y (т.к. по предположению нулевой гипотезы независимы)
    # ну и Вероятности это количество реализаций на общее количество (во всех 4х комбинациях)
    # D в свою очередь подчиняется распределению Хи-квадрат с (n-1)(m-1) степенями свободы

    # Данные будут распределены полиномиально
    n = 255
    m = 391
    N = 646
    x_positive = 90
    y_positive = 84
    x_negative = n - x_positive
    y_negative = m - y_positive

    p_i = np.array([
        n/N * (x_positive + y_positive)/N,
        n/N * (x_negative + y_negative)/N,
        m/N * (x_positive + y_positive)/N,
        m/N * (x_negative + y_negative)/N,
    ])

    E_i = np.array([elem * N for elem in p_i])

    n_i = np.array([
        x_positive,
        n - x_positive,
        y_positive,
        m - y_positive
    ])

    D = ((n_i - E_i)**2 / E_i).sum()
    df = (2-1)*(2-1)

    alpha = 0.05
    p_value = 1 - dist.chi2(df).cdf(D)

    print(alpha, p_value)
    print(alpha < p_value)


def main():
    pass


if __name__ == '__main__':
    # 8-9 лаба

    # Простой односторонний асимптотический
    # simple_1side_assimp()

    # Простой двусторонний асимптотический
    # simple_2side_assimp()

    # На основе нормального с известной общей дисперсией
    # normal_base_known_general_variation()

    # Асимптотический На основе нормального с неизвестной общей дисперсией
    # assimp_normal_base_unknown_general_variation()

    # Точный На основе нормального с неизвестной общей дисперсией
    # exact_normal_base_unknown_general_variation()

    # Т-тест Стьюдента
    # students_t_test()

    # На основе нормального с неизвестной разной дисперсией и неравными выборками
    # normal_base_unknown_different_variation()

    # 10 лаб

    # Биномиальный асимптотический двусторонний на одной выборке
    # binomial_assimp_2side_1sample()

    # Биномиальный точный двусторонний на одной выборке
    # binomial_exact_2side_1sample()

    # Биномиальный асимптотический двусторонний с двумя выборками (АВ-тест)
    # binomial_assimp_1side_2sample()

    # Биномиальный точный двусторонний с двумя выборками (точный Тест Фишера)
    # binomial_exact_2side_2sample()

    # 11 лаба

    # Тест на одинаковость рампределений
    # equals_distributions()

    # Тест на зависимость
    # dependence_test()

    main()
