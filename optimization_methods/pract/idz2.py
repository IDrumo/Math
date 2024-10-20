import numpy as np

materials_directory_path = 'materials/system_state.npz'


# Генерация положительно определенной матрицы A с целыми числами
def generate_positive_definite_matrix(size):
    # Генерируем случайную матрицу с целыми числами и умножаем её на её транспонированную
    A = np.random.randint(0, 3, (size, size))  # Генерация целых чисел от 0 до 4
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
def gradient_descent(A, b, x0, learning_rate=1, epsilon=1e-6, max_iterations=1000):
    x = x0

    for _ in range(max_iterations):
        grad = gradient_f_0(x, A, b)

        # Проверка условия остановки по норме градиента
        if np.linalg.norm(grad) < epsilon:
            break

        x_new = x - learning_rate * grad
        f_new = f_0(x_new, A, b)

        # Коррекция шага
        if f_new >= f_0(x, A, b):
            learning_rate /= 2  # Делим шаг пополам

        # Обновляем значения
        x = x_new

    return x


# Основной код
if __name__ == "__main__":
    size = 3
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
    x_min = gradient_descent(A_loaded, b_loaded, x0)

    print("Минимум функции f_0(x) достигается в точке:", x_min)
    print("Значение функции в этой точке:", f_0(x_min, A_loaded, b_loaded))
