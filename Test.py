from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# генерируем данные
x = [0, 1, 2, 3]
y = [0, 1, 0, 3]
x = np.array(x)
y = np.array(y)
numb=np.arange(0,len(x),1)


#линия регрессии
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

line = slope*x+intercept

#создание кроссплота
fig = plt.figure(figsize=(10,7))
ax = plt.subplot(111)

plt.scatter(x,y, s=50, c=numb)
plt.plot(x, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
plt.plot([], [], ' ', label='R_sq = '+'{:.2f}'.format(r_value**2))

plt.grid(True)
plt.legend(fontsize=12)
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
