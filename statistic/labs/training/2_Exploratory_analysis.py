import numpy as np
import pandas as pd
import scipy.stats as dist
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 12]


def main():
    # Шаг 1: Загрузка и первичный осмотр данных
    data = pd.read_csv('../data/анализ крови.csv')
    # print(data.head()) # Первые 5 строчек
    # print(data.dtypes) # Типы данных
    # print(data.describe()) # описать базу
    # print(data.isnull().sum()) # Проверить на null и NaN

    # Шаг 2: Обработка пропущенных значений
    data2 = data.iloc[:, 1:].fillna(data.iloc[:, 1:].mean())  # Заменим null и NaN на средние значения

    # Шаг 3: Анализ распределения переменных
    data.hist()
    # plt.show()
    plt.clf()

    # Шаг 4: Исследование корреляций между переменными
    sns.heatmap(data2.corr(),
                xticklabels=data2.corr().columns,
                yticklabels=data2.corr().columns)
    # plt.show()
    plt.clf()

    # Шаг 5: Выявление выбросов и аномалий
    plt.close('all') # закрываем все холсты
    fig, axs = plt.subplots(1, len(data2.columns), figsize=(12, 12))
    for i, ax in enumerate(axs.flat):
        ax.boxplot(data2.iloc[:, i]) # Запихиваем коробку с усами на полотно.
        ax.set_title(data2.columns[i])
    plt.tight_layout()
    # plt.show()
    plt.clf()
    plt.close('all')


if __name__ == '__main__':
    main()
