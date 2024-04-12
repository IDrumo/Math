import matplotlib.pyplot as plt
import sympy as sp


def paint_function(expression, start=-5, end=5):
    x = sp.Symbol('x')
    f = sp.lambdify(x, expression, 'numpy')

    points_number = 20
    step = (end - start) / points_number
    x_vals = [start + i * step for i in range(points_number)]
    y_vals = [f(start + i * step) for i in range(points_number)]
    # current_x = start
    # while current_x <= end:
    #     x_vals.append(current_x)
    #     y_vals.append(f(current_x))
    #     current_x += step

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=str(expression))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graph of ' + str(expression))
    plt.grid(True)
    plt.legend()
    plt.show()
