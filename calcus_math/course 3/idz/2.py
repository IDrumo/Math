import numpy as np

for _ in range(50):
    n = np.random.randint(2, 10)  # Случайный размер матрицы
    A = np.random.rand(n, n)      # Случайная матрица
    a = np.random.rand() + 1    # Случайное число

    # Вычисление числа обусловленности
    cond_A = np.linalg.cond(A)
    cond_aA = np.linalg.cond(a * A)

    # Проверка равенства
    assert np.isclose(cond_A, cond_aA)