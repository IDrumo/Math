import matplotlib.pyplot as plt
import numpy as np


def simple_iteration(A, b, x0=None, eps=1e-10, max_iter=1000, tau=None):
    """
    Решение СЛАУ методом простой итерации.

    Параметры:
    A (np.ndarray): Матрица коэффициентов системы.
    b (np.ndarray): Вектор свободных членов.
    x0 (np.ndarray): Начальное приближение решения.
    eps (float): Желаемая точность решения.
    max_iter (int): Максимальное число итераций.

    Возвращает:
    np.ndarray: Решение СЛАУ.
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)

    x = np.copy(x0)

    if tau is None:
        # Вычисление собственных значений для определения tau
        eigenvalues = np.linalg.eigvals(A)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        min_eigenvalue = np.min(np.abs(eigenvalues))

        tau = 2 / (max_eigenvalue + min_eigenvalue)

    for iteration in range(max_iter):
        # Вычисление нового приближения
        x_new = x + tau * (b - np.dot(A, x))

        if np.linalg.norm(x_new - x) < eps:
            return x_new, iteration

        x = x_new

    # print(f"Метод простой итерации не сошелся за {max_iter} итераций.")
    return x, max_iter


def plot_iterations(x_values, y_values, x_label, y_label, title, highlight_x=None, highlight_y=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', label='Количество итераций')
    if highlight_x is not None and highlight_y is not None:
        plt.plot(highlight_x, highlight_y, 'ro', label='Оптимальное значение')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend()
    plt.show()


def generate_conditioned_matrix(n, condition_number):
    """Генерирует матрицу с заданным числом обусловленности."""
    # Генерируем случайную матрицу
    Q, _ = np.linalg.qr(np.random.rand(n, n))  # QR-разложение для получения ортогональной матрицы
    # Создаем диагональную матрицу с заданными собственными значениями
    singular_values = np.linspace(1, condition_number, n)
    S = np.diag(singular_values)
    # Возвращаем матрицу с заданным числом обусловленности
    return Q @ S @ Q.T


def plot_iterations_vs_size(max_size=20, condition_number=10):
    sizes = range(2, max_size + 1)
    iterations = []

    for n in sizes:
        A = generate_conditioned_matrix(n, condition_number)  # Генерируем матрицу с заданным числом обусловленности
        b = np.random.rand(n)
        _, iter_count = simple_iteration(A, b)
        iterations.append(iter_count)

    plot_iterations(sizes, iterations, 'Размерность матрицы (n)', 'Количество итераций',
                    'Зависимость числа итераций от размерности матрицы')


def plot_iterations_vs_eps():
    epsilons = [10 ** (-i) for i in range(1, 11)]
    iterations = []

    for eps in epsilons:
        A = np.array([[4, -1, 0, 0],
                      [-1, 4, -1, 0],
                      [0, -1, 4, -1],
                      [0, 0, -1, 3]], dtype=float)
        b = np.array([15, 10, 10, 10], dtype=float)
        _, iter_count = simple_iteration(A, b, eps=eps)
        iterations.append(iter_count)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, iterations, marker='o', label='Количество итераций')
    plt.title('Зависимость числа итераций от значения эпсилон')
    plt.xlabel('Эпсилон')
    plt.ylabel('Количество итераций')
    plt.xscale('log')  # Устанавливаем логарифмическую шкалу для оси X
    plt.grid()
    plt.legend()
    plt.show()


def plot_iterations_vs_tau(max_tau=0.4, step=0.01):
    taus = np.arange(0.001, max_tau, step)
    iterations = []

    A = np.array([[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 3]], dtype=float)
    b = np.array([15, 10, 10, 10], dtype=float)

    for tau in taus:
        _, iter_count = simple_iteration(A, b, tau=tau)
        iterations.append(iter_count)

    # Находим оптимальное значение тау
    eigenvalues = np.linalg.eigvals(A)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    min_eigenvalue = np.min(np.abs(eigenvalues))
    optimal_tau = 2 / (max_eigenvalue + min_eigenvalue)
    _, optimal_iterations = simple_iteration(A, b, tau=optimal_tau)

    # optimal_index = np.argmin(iterations)
    # optimal_tau = taus[optimal_index]
    # optimal_iterations = iterations[optimal_index]

    plot_iterations(taus, iterations, 'Тау', 'Количество итераций',
                    'Зависимость числа итераций от значения тау',
                    highlight_x=optimal_tau, highlight_y=optimal_iterations)


if __name__ == '__main__':
    A = np.array([[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 3]], dtype=float)
    b = np.array([15, 10, 10, 10], dtype=float)
    x0 = np.zeros(len(b))

    solution, iterations = simple_iteration(A, b, x0)
    print("Решение:", solution)
    print("Количество итераций:", iterations)
    print("Невязка:", np.linalg.norm(A @ solution - b))

    plot_iterations_vs_size()
    plot_iterations_vs_eps()
    plot_iterations_vs_tau()
