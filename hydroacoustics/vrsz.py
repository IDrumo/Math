import numpy as np
import matplotlib.pyplot as plt

# Данные ВРСЗ для варианта №19
depths = np.array([0, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 800])
speeds = np.array([1497, 1494, 1493, 1492, 1490, 1492, 1493, 1495, 1499, 1503, 1507, 1511, 1519, 1522, 1533])


# Построение графика
plt.figure(figsize=(14, 8))
plt.plot(speeds, depths, marker='o', linestyle='-')
plt.gca().invert_yaxis()  # инвертировать ось глубины
plt.title("Вертикальное распределение скорости звука (ВРСЗ)\nВариант №23")
plt.xlabel("Скорость звука, м/с")
plt.ylabel("Глубина, м")
plt.grid(True)
plt.tight_layout()
plt.show()
