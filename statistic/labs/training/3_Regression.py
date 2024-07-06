import numpy as np
import pandas as pd
import scipy.stats as dist
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 12]


def main():
    data = pd.read_csv('../data/анализ крови.csv')

    data2 = data.iloc[:, 1:].fillna(data.iloc[:, 1:].mean())  # Заменим null и NaN на средние значения
    data3 = data.iloc[:, 1:].dropna()

    dater = data2
    sns.heatmap(dater.corr(),
                xticklabels=dater.corr().columns,
                yticklabels=dater.corr().columns)
    # plt.show()
    plt.clf()
    plt.close('all')

    clear_data = dater[(dater['density'] <= 1.01)]  # почистил от выбросов
    dater = clear_data
    clear_data.plot.scatter(x='density', y='alcohol')
    # plt.show()
    plt.clf()
    plt.close('all')

    info = sm.OLS.from_formula('density ~ alcohol', dater).fit()
    # Method - показывает метод, который был использован для построения регрессии (Метод наименьших квадратов по умолчанию)
    # R-squared и Adj. R-squared - это коэф. от 0 до 1, который показывает как хорошо данные укладываются в модель
    # Как посчитать?
    # R = 1 - ((clear_data['alcohol'] - regress_func(x))**2).sum() /((clear_data['alcohol'] - clear_data['alcohol'].mean())**2).sum()
    # beta_0 = clear_data[['density', 'alcohol']].corr()[0, 1] * clear_data['alcohol'].std() / clear_data['density'].std()
    # Prob (F-statistic) - вероятность того, что наш коэффициент равен нулю
    # coef содержит коэф. регрессии: Intercept - свободный коэф. alcohol - неизв. при переменной
    # std err - стандартная ошибка параметров
    # t - Т-статистика
    # P>|t| - с какой вероятностью коэф. равен нулю. Если с высокой, то скорее всего исследуемый параметр не зависит от этого
    # [0.025      0.975] - 95%-ый интервал значений
    # print(info.summary()) # Выводит инфу о линейной регрессии
    sns.regplot(x='density', y='alcohol', data=clear_data)
    # plt.show()
    plt.clf()
    plt.close('all')

    # Регрессия для всего (многомерная)
    # print(clear_data.columns)
    dater.columns = dater.columns.str.replace(' ', '_')
    info = sm.OLS.from_formula(
        "alcohol ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + quality",
        dater).fit()
    print(info.summary())

    # Далее то же самое ручками:
    n = len(dater['alcohol'])
    X = dater.drop(columns='alcohol').to_numpy()  # сконвертировали матрицу в numpy массив
    X = np.c_[np.ones(n), X]  # Добавили столбец единиц (интерцепт)
    Y = dater['alcohol'].to_numpy()
    Betta = np.linalg.inv(X.T @ X) @ X.T @ Y
    print(Betta)


if __name__ == '__main__':
    main()
