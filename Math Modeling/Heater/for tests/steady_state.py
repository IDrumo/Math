import matplotlib.pyplot as plt
import numpy as np

# Параметры модели
P = 3000
k = 2
S = 0.4
sigma = 5.67e-8
T0 = 296

# Функция для поиска корней
def f(T):
    return P - k*S*(T - T0) - sigma*S*(T**4 - T0**4)

# Построение графика
T_range = np.linspace(250, 650, 1000)
plt.figure(figsize=(8, 5))
plt.plot(T_range, f(T_range), 'b-', label='$f(T)$')
plt.axhline(0, color='k', linestyle='--')
plt.axvline(599.58, color='r', linestyle=':', label='$T_{\mathrm{eq}}$')
plt.xlabel('Температура, $T$ (К)')
plt.ylabel('$f(T)$')
plt.title('График функции теплового баланса')
plt.grid(True)
plt.legend()
plt.savefig('steady_state.png', dpi=300, bbox_inches='tight')