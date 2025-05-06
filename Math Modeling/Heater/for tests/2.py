import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры системы
P = 3000  # Вт
k = 2  # Вт/(м²·К)
S = 0.4  # м²
sigma = 5.67e-8  # Вт/(м²·К⁴)
T0 = 296  # К

def dTdt_controlled(t, T, m, c, T_max, T_min):
    # Функция управления
    if T > T_max:
        H = 0
    elif T < T_min:
        H = 1
    else:
        H = H_prev
    return (P * H - k * S * (T - T0) - sigma * S * (T**4 - T0**4)) / (c * m)

# Начальные условия и время интегрирования
t_span = (0, 1000)  # с
T_init = T0  # К


# Параметры для разных кривых
params_controlled = [
    {"T_max": 550, "T_min": 500, "label": "$T_{\\text{max}}=550$ K, $T_{\\text{min}}=500$ K", "color": "blue"},
    {"T_max": 600, "T_min": 550, "label": "$T_{\\text{max}}=600$ K, $T_{\\text{min}}=550$ K", "color": "red"},
]

# Решение ОДУ и построение графиков
plt.figure(figsize=(12, 6))
for i, p in enumerate(params_controlled):
    sol = solve_ivp(dTdt_controlled, t_span, [T_init], args=(0.4, 554, p["T_max"], p["T_min"]), max_step=1)
    plt.plot(sol.t, sol.y[0], color=p["color"], label=p["label"])
    plt.axhline(p["T_max"], color="gray", linestyle="--")
    plt.axhline(p["T_min"], color="gray", linestyle="--")

# Оформление графика
plt.xlabel("Время, $t$ (с)")
plt.ylabel("Температура, $T$ (К)")
plt.title("Модель с терморегулятором")
plt.grid(True)
plt.legend()
plt.savefig("model_control.png", bbox_inches="tight")
plt.show()