import matplotlib.pyplot as plt
import numpy as np


def hurwitz_criterion(coefficients):
    n = len(coefficients) - 1
    if n < 1:
        return False

    # Создаем матрицу Гурвица
    hurwitz = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Индекс в списке коэффициентов
            index = 2 * i - j + 1
            if 0 <= index < len(coefficients):
                hurwitz[i, j] = coefficients[index]

    # Проверяем определитель главных миноров
    for k in range(1, n + 1):
        if np.linalg.det(hurwitz[:k, :k]) <= 0:
            return False

    return True


def mikhailov_criterion(coefficients):
    # Получаем корни характеристического уравнения
    roots = np.roots(coefficients)

    # Проверяем, находятся ли корни в левой полуплоскости
    return all(np.real(roots) < 0)


def visualize_mikhailov(coefficients):
    # Получаем корни характеристического уравнения
    roots = np.roots(coefficients)

    # Визуализация корней в комплексной плоскости
    plt.figure(figsize=(10, 8))

    # Генерация случайных цветов для каждого корня
    colors = np.random.rand(len(roots), 3)  # RGB цвета

    for i, root in enumerate(roots):
        plt.scatter(np.real(root), np.imag(root), color=colors[i], marker='o')
        plt.text(np.real(root), np.imag(root), f'{root:.2f}', fontsize=10, ha='right', color=colors[i])

    # Отметим оси
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')

    # Настройки графика
    plt.title('Корни характеристического уравнения')
    plt.xlabel('Действительная часть')
    plt.ylabel('Мнимая часть')
    plt.xlim(-2, 1)
    plt.ylim(-2, 2)
    plt.grid()

    plt.show()


def lienard_shipar_criterion(coefficients):
    # Получаем корни характеристического уравнения
    roots = np.roots(coefficients)
    real_parts = np.real(roots)

    # Проверяем условия устойчивости
    return all(real_parts < 0)


# Пример использования
coeffs = [1, 2, 2, 3]

print("Критерий Гурвица:", hurwitz_criterion(coeffs))
print("Критерий Михайлова:", mikhailov_criterion(coeffs))
print("Критерий Льенара-Шипара:", lienard_shipar_criterion(coeffs))

# Визуализация корней
visualize_mikhailov(coeffs)
