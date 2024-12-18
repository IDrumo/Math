import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SurfaceVisualizer:
    def __init__(self, A, b):
        """
        Инициализация класса.

        :param A: Матрица коэффициентов (numpy array)
        :param b: Вектор свободных членов (numpy array)
        """
        self.A = np.array(A)
        self.b = np.array(b)

    def visualize_surface(self):
        """Визуализирует поверхность, заданную СЛАУ."""
        if self.A.shape[0] == 1 and self.A.shape[1] == 2:
            self.visualize_2d_surface()
        elif self.A.shape[0] == 1 and self.A.shape[1] == 3:
            self.visualize_3d_surface()
        elif self.A.shape[0] == 2 and self.A.shape[1] == 3:
            self.visualize_3d_surface_multiple()
        else:
            print("Поддерживаются только 2D и 3D поверхности.")

    def visualize_2d_surface(self):
        """Визуализация 2D поверхности."""
        # Уравнение вида: a1*x + b1*y = c1
        a1, b1 = self.A[0]
        c1 = self.b[0]

        x = np.linspace(-10, 10, 100)
        y = (c1 - a1 * x) / b1

        plt.figure()
        plt.plot(x, y, label=f'{a1}x + {b1}y = {c1}')
        plt.title('2D Surface Visualization')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.axhline(0, color='black', linewidth=0.5, ls='--')
        plt.axvline(0, color='black', linewidth=0.5, ls='--')
        plt.grid()
        plt.legend()
        plt.show()

    def visualize_3d_surface(self):
        """Визуализация 3D поверхности."""
        # Уравнение вида: a1*x + b1*y + c1*z = d1
        a1, b1, c1 = self.A[0]
        d1 = self.b[0]

        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)

        Z = (d1 - a1 * X - b1 * Y) / c1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_title('3D Surface Visualization')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.show()

    def visualize_3d_surface_multiple(self):
        """Визуализация нескольких 3D поверхностей."""
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(self.A.shape[0]):
            a1, b1, c1 = self.A[i]
            d1 = self.b[i]
            Z = (d1 - a1 * X - b1 * Y) / c1
            ax.plot_surface(X, Y, Z, alpha=0.5, edgecolor='none')

        ax.set_title('3D Surface Visualization (Multiple)')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.show()

