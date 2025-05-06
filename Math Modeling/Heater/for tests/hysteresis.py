import matplotlib.pyplot as plt
import numpy as np

# Параметры модели
T_min = 300  # К
T_max = 350  # К
P = 1000     # Вт
cm = 5000    # Дж/К
k = 10       # Вт/(м²·К)
S = 0.5      # м²
sigma = 5.67e-8
T0 = 298     # К

# Решение дифференциального уравнения с терморегулятором
def dTdt(t, T):
    H = 1 if (T < T_min) else (0 if (T > T_max) else H_prev)
    return (P * H - k * S * (T - T0) - sigma * S * (T**4 - T0**4)) / cm

# Численное интегрирование (метод Эйлера)
dt = 1  # шаг времени (с)
t = np.arange(0, 1000, dt)
T = np.zeros_like(t)
H = np.zeros_like(t)
T[0] = T0
H_prev = 1  # начальное состояние

for i in range(1, len(t)):
    H_prev = 1 if (T[i-1] < T_min) else (0 if (T[i-1] > T_max) else H_prev)
    H[i] = H_prev
    T[i] = T[i-1] + dTdt(t[i-1], T[i-1]) * dt

import matplotlib.pyplot as plt
import numpy as np

# Параметры модели (те же, что выше)
T_range = np.linspace(250, 400, 500)

# Вычисление dT/dt для H=1 и H=0
dTdt_on = [(P - k*S*(T-T0) - sigma*S*(T**4 - T0**4)) / cm for T in T_range]
dTdt_off = [(-k*S*(T-T0) - sigma*S*(T**4 - T0**4)) / cm for T in T_range]

# Построение
plt.figure(figsize=(8, 5))
plt.plot(T_range, dTdt_on, 'g-', label='$H=1$ (нагрев)')
plt.plot(T_range, dTdt_off, 'r-', label='$H=0$ (охлаждение)')

# Области гистерезиса
plt.fill_betweenx([-0.5, 2], T_min, T_max, color='gray', alpha=0.2, label='Зона гистерезиса')

# Оформление
plt.axvline(T_min, color='b', linestyle=':', label='$T_{\mathrm{min}}$')
plt.axvline(T_max, color='r', linestyle=':', label='$T_{\mathrm{max}}$')
plt.title('Фазовая диаграмма системы с гистерезисом')
plt.xlabel('Температура, $T$ (К)')
plt.ylabel('Скорость изменения $dT/dt$ (К/с)')
plt.ylim(-0.5, 2)
plt.grid(True)
plt.legend()
plt.savefig('phase_diagram.png', dpi=300, bbox_inches='tight')