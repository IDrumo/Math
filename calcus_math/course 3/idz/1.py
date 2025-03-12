import numpy as np

for _ in range(50):
    n = np.random.randint(2, 10)  # Случайный размер матрицы
    A = np.random.rand(n, n)  # Случайная матрица

    # Вычисление норм
    inf_norm = np.linalg.norm(A, np.inf)
    m_norm = n * np.max(np.abs(A))

    # Проверка соотношения
    assert inf_norm <= m_norm <= n * inf_norm
