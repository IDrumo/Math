import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры системы
P = 3000  # Вт
k = 2  # Вт/(м²·К)
S = 0.4  # м²
sigma = 5.67e-8  # Вт/(м²·К⁴)
T0 = 296  # К

# Функция правой части ОДУ
def dTdt(t, T, m, c):
    return (P - k * S * (T - T0) - sigma * S * (T**4 - T0**4)) / (c * m)

# Начальные условия и время интегрирования
t_span = (0, 1000)  # с
T_init = T0  # К

# Параметры для разных кривых
params = [
    {"m": 0.5, "c": 897, "label": "m=0.5 кг, c=897 Дж/(кг·К)", "color": "blue"},
    {"m": 1.0, "c": 897, "label": "m=1.0 кг, c=897 Дж/(кг·К)", "color": "red"},
    {"m": 0.5, "c": 450, "label": "m=0.5 кг, c=450 Дж/(кг·К)", "color": "green"},
]

# Решение ОДУ и построение графиков
plt.figure(figsize=(10, 6))
for p in params:
    sol = solve_ivp(dTdt, t_span, [T_init], args=(p["m"], p["c"]), max_step=1)
    plt.plot(sol.t, sol.y[0], color=p["color"], label=p["label"])

# Оформление графика
plt.axhline(599.58, color="black", linestyle="--", label="Теоретическая точка равновесия")
plt.xlabel("Время, $t$ (с)")
plt.ylabel("Температура, $T$ (К)")
plt.title("Модель без терморегулятора")
plt.grid(True)
plt.legend()
plt.savefig("model_base.png", bbox_inches="tight")
plt.show()