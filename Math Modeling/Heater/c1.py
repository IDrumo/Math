import numpy as np
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

# Пример использования
P = 3000  # Мощность, Вт
m = 0.5  # Масса, кг
c = 897  # Удельная теплоёмкость, Дж/(кг·К)
S = 0.4  # Площадь поверхности, м²
k = 2  # Коэффициент теплоотдачи, Вт/(м²·К)

# Решение ОДУ для модели без регулятора
sol_base = solve_ivp(dTdt, t_span, [T_init], args=(P, m, c, S, k), max_step=1)

# Решение ОДУ для модели с регулятором
T_max = 550  # Максимальная температура, К
T_min = 500  # Минимальная температура, К
sol_controlled = solve_ivp(dTdt_controlled, t_span, [T_init], args=(P, m, c, S, k, T_max, T_min), max_step=1)