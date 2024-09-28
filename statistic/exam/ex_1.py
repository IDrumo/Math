import numpy as np
import pandas as pd
import scipy.stats as dist
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 12]


def corr(X, Y):
    x_mean = X.sum() / len(X)
    y_mean = Y.sum() / len(Y)
    xy_mean = (X * Y).sum() / len(X)
    sigma_x = np.sqrt(((X - x_mean) ** 2).sum() / len(X))
    sigma_y = np.sqrt(((Y - y_mean) ** 2).sum() / len(Y))

    correlation = (xy_mean - x_mean * y_mean) / (sigma_x * sigma_y)
    return correlation


def main():
    # Шаг 1: Загрузка и первичный осмотр данных
    data = pd.read_csv('data/Social_Network_Ads.csv', index_col=0)
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

    print(data.head())  # Первые 5 строчек

    print(data.dtypes)  # Типы данных

    print(data.describe())  # описать базу

    print(data.isnull().sum())  # Проверить на null и NaN

    # print(f"стандартное отклонение: {np.sqrt(((data['Purchased'].to_numpy() - data['Purchased'].mean())**2).sum() / data['Purchased'].size)}")

    # Шаг 2: Анализ распределения переменных
    data.hist(bins=100)
    # plt.show()
    plt.clf()

    # Шаг 3: Исследование корреляций между переменными
    sns.heatmap(data.corr(),
                xticklabels=data.corr().columns,
                yticklabels=data.corr().columns)
    plt.show()
    plt.clf()

    # Матрица корреляций:
    print(data.corr())
    # print(data.corr()['Purchased'])

    print(corr(data['Age'], data['EstimatedSalary']))

    # Шаг 4: Выявление выбросов и аномалий

    plt.close('all')  # закрываем все холсты
    fig, axs = plt.subplots(1, len(data.columns), figsize=(12, 12))
    for i, ax in enumerate(axs.flat):
        ax.boxplot(data.iloc[:, i])  # Запихиваем коробку с усами на полотно.
        ax.set_title(data.columns[i])
    plt.tight_layout()
    # plt.show()
    plt.clf()
    plt.close('all')


if __name__ == '__main__':
    main()
