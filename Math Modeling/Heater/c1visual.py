import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры системы
sigma = 5.67e-8  # Постоянная Стефана-Больцмана, Вт/(м²·К⁴)
T0 = 296  # Начальная температура, К

# Функция правой части ОДУ для модели без регулятора
def dTdt(t, T, P, m, c, S, k):
    return (P - k * S * (T - T0) - sigma * S * (T**4 - T0**4)) / (c * m)

# Функция правой части ОДУ для модели с регулятором
def dTdt_controlled(t, T, P, m, c, S, k, T_max, T_min):
    global H_prev
    if T > T_max:
        H = 0
    elif T < T_min:
        H = 1
    else:
        H = H_prev
    H_prev = H
    return (P * H - k * S * (T - T0) - sigma * S * (T**4 - T0**4)) / (c * m)

# Начальные условия и время интегрирования
t_span = (0, 500)  # Время интегрирования, с
T_init = T0  # Начальная температура, К
H_prev = 1  # Начальное состояние регулятора

# Параметры для модели без регулятора
params_base = [
    {"P": 3000, "m": 0.5, "c": 897, "S": 0.4, "k": 2, "label": "Алюминий", "color": "#1f77b4"},
    {"P": 3000, "m": 1.0, "c": 897, "S": 0.4, "k": 2, "label": "Алюминий (2x масса)", "color": "#ff7f0e"},
    {"P": 3000, "m": 0.5, "c": 450, "S": 0.4, "k": 2, "label": "Сталь", "color": "#2ca02c"},
    {"P": 3000, "m": 0.5, "c": 897, "S": 0.2, "k": 2, "label": "Уменьшенная площадь", "color": "#d62728"},
]

# Параметры для модели с регулятором
params_controlled = [
    {"P": 3000, "m": 0.5, "c": 897, "S": 0.4, "k": 2, "T_max": 550, "T_min": 500, "label": "$T_{\\text{max}}=550$ K, $T_{\\text{min}}=500$ K", "color": "#1f77b4"},
    {"P": 3000, "m": 0.5, "c": 897, "S": 0.4, "k": 2, "T_max": 600, "T_min": 550, "label": "$T_{\\text{max}}=600$ K, $T_{\\text{min}}=550$ K", "color": "#ff7f0e"},
    {"P": 3000, "m": 0.5, "c": 897, "S": 0.4, "k": 2, "T_max": 500, "T_min": 450, "label": "$T_{\\text{max}}=500$ K, $T_{\\text{min}}=450$ K", "color": "#2ca02c"},
    {"P": 3000, "m": 0.5, "c": 897, "S": 0.4, "k": 2, "T_max": 650, "T_min": 600, "label": "$T_{\\text{max}}=650$ K, $T_{\\text{min}}=600$ K", "color": "#d62728"},
]

# Решение и визуализация для модели без регулятора
plt.figure(figsize=(10, 6))
for p in params_base:
    sol = solve_ivp(dTdt, t_span, [T_init], args=(p["P"], p["m"], p["c"], p["S"], p["k"]), max_step=1)
    plt.plot(sol.t, sol.y[0], color=p["color"], label=p["label"])

plt.axhline(599.58, color="black", linestyle="--", label="Теоретическая точка равновесия")
plt.xlabel("Время, $t$ (с)")
plt.ylabel("Температура, $T$ (К)")
plt.title("Модель без терморегулятора")
plt.grid(True)
plt.legend()
plt.savefig("model_base.pdf", bbox_inches="tight")
plt.show()

# Решение и визуализация для модели с регулятором
plt.figure(figsize=(10, 6))
for p in params_controlled:
    sol = solve_ivp(dTdt_controlled, t_span, [T_init], args=(p["P"], p["m"], p["c"], p["S"], p["k"], p["T_max"], p["T_min"]), max_step=1)
    plt.plot(sol.t, sol.y[0], color=p["color"], label=p["label"])
    plt.axhline(p["T_max"], color="gray", linestyle="--")
    plt.axhline(p["T_min"], color="gray", linestyle="--")

plt.xlabel("Время, $t$ (с)")
plt.ylabel("Температура, $T$ (К)")
plt.title("Модель с терморегулятором")
plt.grid(True)
plt.legend()
plt.savefig("model_control.pdf", bbox_inches="tight")
plt.show()