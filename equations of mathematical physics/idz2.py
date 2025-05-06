import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Определение функции f(x)
def f(x):
    return (x * np.cos(np.pi * x / 1.8))**2

# Определение собственных функций y_k
def y_k(x, k):
    return np.sin((5 * np.pi + 10 * np.pi * k) * x / 9)

# Вычисление коэффициента c_k
def compute_c_k(k):
    integral_num, _ = quad(lambda x: f(x) * y_k(x, k), 0, 0.9)
    norm_y_k = 9 / 20
    return integral_num / norm_y_k

horm_num = 5
# Вычисление коэффициентов c_k для k=0,1,2,3,4
c = []
for k in range(horm_num):
    ck = compute_c_k(k)
    c.append(ck)
    print(f'c_{k} = {ck:.5f}')

# Построение графиков
x = np.linspace(0, 0.9, 1000)
f_x = f(x)

# Вычисление суммы ряда Фурье с пятью гармониками
fourier_sum = np.zeros_like(x)
for k in range(horm_num):
    fourier_sum += c[k] * y_k(x, k)

# Графики
plt.figure(figsize=(10, 6))
plt.plot(x, f_x, label='f(x)')
plt.plot(x, fourier_sum, label=f'Ряд Фурье ({horm_num} гармоник)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Функция и её разложение в ряд Фурье')
plt.legend()
plt.grid(True)
# plt.show()

# Проверка равенства Парсеваля
integral_f_squared, _ = quad(lambda x: f(x)**2, 0, 0.9)
sum_c_squared = 9/20 * sum(ck**2 for ck in [compute_c_k(k) for k in range(2)])

print(f'\nИнтеграл от f(x)^2: {integral_f_squared:.5f}')
print(f'Сумма квадратов c_k: {sum_c_squared:.5f}')
print(f'Разница: {abs(integral_f_squared - sum_c_squared):.5f}')

# Проверка точности
if abs(integral_f_squared - sum_c_squared) < 1e-3:
    print("Равенство Парсеваля выполняется с точностью до 1e-3")
else:
    print("Равенство Парсеваля не выполняется с требуемой точностью")