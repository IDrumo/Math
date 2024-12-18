import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, step

# Определение передаточной функции
num = [1]  # Числитель
den = [1, 3, 2]  # Знаменатель
system = TransferFunction(num, den)

# Вычисление корней характеристического уравнения
roots = np.roots(den)

# Визуализация корней
plt.figure(figsize=(8, 6))
plt.scatter(roots.real, roots.imag, color='red', marker='x')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.title('Корни характеристического уравнения')
plt.xlabel('Действительная часть')
plt.ylabel('Мнимая часть')
plt.grid()
plt.xlim(-5, 1)
plt.ylim(-3, 3)
plt.show()

# Визуализация переходной характеристики
t, y = step(system)
plt.figure(figsize=(8, 6))
plt.plot(t, y)
plt.title('Переходная характеристика системы')
plt.xlabel('Время (s)')
plt.ylabel('Амплитуда')
plt.grid()
plt.show()