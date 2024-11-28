import numpy as np
from scipy.optimize import minimize

# materials_directory_path = '../../optimization_methods/pract/materials/system_state.npz'
materials_directory_path = 'materials/system_state.npz'


# Генерация положительно определенной матрицы A с целыми числами
def generate_positive_definite_matrix(size):
    A = np.random.randint(0, 3, (size, size))  # Генерация целых чисел от 0 до 3
    A = np.dot(A, A.T)  # Положительно определенная матрица
    return A


# Сохранение матрицы A и вектора b в один файл
def save_system_state(A, b, filename):
    np.savez(filename, A=A, b=b)


# Считывание матрицы A и вектора b из файла
def load_system_state(filename):
    data = np.load(filename)
    return data['A'], data['b']


# Отображение матрицы и вектора
def display_matrix_and_vector(A, b):
    print("Сохраненная матрица A:")
    print(A)
    print("Сохраненный вектор b:")
    print(b)


# Функция f_0(x)
def f_0(x, A, b):
    return 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b, x)


# Градиент функции f_0(x)
def gradient_f_0(x, A, b):
    return np.dot(A, x) + b


# Метод градиентного спуска
def mu(r, A):
    """
    Функция для вычисления параметра mu, который используется для обновления решения.
    r - остаток (разность между Ax и b)
    A - матрица
    """
    return np.dot(r, A @ r) / np.dot(A @ r, A @ r)


def gradient_descent(A, b, x0, eps=1e-10, max_iterations=1000):
    x = x0

    for i in range(max_iterations):
        r_k = np.dot(A, x) + b  # Остаток
        mu_k = mu(r_k, A)  # Вычисляем mu

        x_new = x - mu_k * gradient_f_0(x, A, b)  # Обновляем x

        # Проверка условия остановки по изменению x
        if np.linalg.norm(x_new - x) < eps:
            return x_new, i  # Возвращаем найденное значение и количество итераций

        x = x_new  # Обновляем x для следующей итерации

    return x, max_iterations  # Если не достигли сходимости, возвращаем последнее значение


# Основной код
if __name__ == "__main__":
    size = 4
    # Генерируем матрицу A
    A = generate_positive_definite_matrix(size)

    # Генерация вектора b с целыми числами
    b = np.random.randint(1, 10, size)  # Генерация целых чисел от 1 до 9

    # Сохраняем состояние системы в файл
    # save_system_state(A, b, materials_directory_path)

    # Считываем состояние системы из файла
    A_loaded, b_loaded = load_system_state(materials_directory_path)

    # Отображаем загруженную матрицу и вектор
    display_matrix_and_vector(A_loaded, b_loaded)

    # Начальная точка
    x0 = np.zeros(size)

    # Находим минимум
    x_min, iterations = gradient_descent(A_loaded, b_loaded, x0, max_iterations=10000)

    print("Минимум функции f_0(x) достигается в точке:", x_min)
    print("За ", iterations, " итераций")
    print("Значение функции в этой точке:", f_0(x_min, A_loaded, b_loaded))
    print()

    # Находим минимум с помощью функции minimize
    result = minimize(f_0, x0, args=(A_loaded, b_loaded), method='BFGS', jac=gradient_f_0)

    if result.success:
        x_min = result.x
        print("Минимум функции f_0(x) достигается в точке:", x_min)
        print("Значение функции в этой точке:", result.fun)
    else:
        print("Не удалось найти минимум функции.")

    print()

    A = np.array([
        [10.9, 1.2, 2.1, 0.9],
        [1.2, 11.2, 1.5, 2.5],
        [2.1, 1.5, 9.8, 1.3],
        [0.9, 2.5, 1.3, 12.1]
    ])

    # Вектор b
    b = np.array([1, 2, 3, 4])  # Пример вектора b

    # Находим минимум с новым значением learning_rate
    x0 = np.zeros(size)
    x_min, iterations = gradient_descent(A, b, x0, max_iterations=10000)

    print("Минимум функции f_0(x) достигается в точке:", x_min)
    print("За ", iterations, " итераций")
    print("Значение функции в этой точке:", f_0(x_min, A, b))
