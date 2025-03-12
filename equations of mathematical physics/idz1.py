import numpy as np
import matplotlib.pyplot as plt

# Параметры
a = 1  # скорость волны
t_values = [1/(4*a), 1/(2*a), 3/(4*a), 1/a, 2/a]  # моменты времени
x = np.linspace(-2, 10, 400)  # диапазон x

# Начальная функция u(x, 0)
def u_initial(x):
    return np.where((x >= 0) & (x <= 8), x**3 * np.exp(-x), 0)

# Функция для вычисления u(x, t)
def u(x, t):
    u_left = u_initial(x - a*t)
    u_right = u_initial(x + a*t)
    return 0.5 * (u_left + u_right)

# Визуализация
plt.figure(figsize=(10, 8))
for t in t_values:
    plt.plot(x, u(x, t), label=f't = {t:.2f}')

plt.title('Профиль бесконечной струны в разные моменты времени')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.xlim(-2, 10)
plt.ylim(0, 2)
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()