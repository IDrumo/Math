import numpy as np
import matplotlib.pyplot as plt
from thermal_model import ThermalSystem

# Общие параметры
common_params = {
    'T0': 296,  # Начальная температура
    'S': 0.4,  # Площадь поверхности
    'k': 2.5  # Коэффициент теплоотдачи
}

# Параметры для 4 моделей без регулятора
models = [
    {'name': 'Медь', 'P': 2800, 'm': 0.8, 'c': 385, 'color': '#1f77b4'},
    {'name': 'Алюминий', 'P': 2500, 'm': 1.2, 'c': 900, 'color': '#ff7f0e'},
    {'name': 'Сталь', 'P': 3000, 'm': 1.5, 'c': 502, 'color': '#2ca02c'},
    {'name': 'Титан', 'P': 3200, 'm': 0.9, 'c': 522, 'color': '#d62728'}
]

# Параметры для 4 моделей с регулятором
control_cases = [
    {'T_max': 550, 'T_min': 500, 'color': '#1f77b4', 'label': 'Случай 1'},
    {'T_max': 390, 'T_min': 360, 'color': '#ff7f0e', 'label': 'Случай 2'},
    {'T_max': 490, 'T_min': 400, 'color': '#2ca02c', 'label': 'Случай 3'},
    {'T_max': 585, 'T_min': 575, 'color': '#d62728', 'label': 'Случай 4'}
]


def plot_uncontrolled():
    plt.figure(figsize=(12, 6))
    for model in models:
        params = {**common_params, **model}
        system = ThermalSystem(params)
        sol = system.solve((0, 1200))
        plt.plot(sol.t, sol.y[0],
                 color=model['color'],
                 label=f"{model['name']}: {params['P']} Вт")

    plt.axhline(623, color='black', linestyle='--', label='Предел нагрева')
    plt.xlabel('Время, с', fontsize=12)
    plt.ylabel('Температура, K', fontsize=12)
    plt.title('Модели без терморегулятора', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.savefig('uncontrolled_models.pdf', bbox_inches='tight')


def plot_controlled():
    plt.figure(figsize=(12, 6))
    base_params = {**common_params, 'P': 3000, 'm': 1.0, 'c': 750}

    for case in control_cases:
        system = ThermalSystem(base_params)
        sol = system.solve((0, 1800), True, (case['T_max'], case['T_min']))
        plt.plot(sol.t, sol.y[0],
                 color=case['color'],
                 label=f"{case['label']}: {case['T_min']}-{case['T_max']}K")

        plt.axhline(case['T_max'], color=case['color'], linestyle=':')
        plt.axhline(case['T_min'], color=case['color'], linestyle=':')

    plt.xlabel('Время, с', fontsize=12)
    plt.ylabel('Температура, K', fontsize=12)
    plt.title('Модели с терморегулятором', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.savefig('controlled_models.pdf', bbox_inches='tight')


if __name__ == "__main__":
    plot_uncontrolled()
    plot_controlled()
    plt.show()