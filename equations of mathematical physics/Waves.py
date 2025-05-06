import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


class WaveEquation:
    def __init__(self, l, a=1.0):
        self.l = l
        self.a = a
        self.phi_k = lambda x, k: np.sin(k * np.pi * x / self.l)

    def set_initial_conditions(self, u0, ut0):
        self.u0 = u0
        self.ut0 = ut0

    def compute_Ak(self, k):
        integrand = lambda x: self.u0(x) * self.phi_k(x, k)
        integral, _ = quad(integrand, 0, self.l)
        return (2 / self.l) * integral

    def compute_Bk(self, k):
        integrand = lambda x: self.ut0(x) * self.phi_k(x, k)
        integral, _ = quad(integrand, 0, self.l)
        return (2 / (self.l * self.a * k * np.pi)) * integral

    def analytical_solution(self, x, t, n_terms=10):
        if isinstance(x, np.ndarray):
            solution = np.zeros_like(x)
            for i, xi in enumerate(x):
                sol = 0.0
                for k in range(1, n_terms + 1):
                    Ak = self.compute_Ak(k)
                    Bk = self.compute_Bk(k)
                    omega = self.a * k * np.pi / self.l
                    sol += (Ak * np.cos(omega * t) + Bk * np.sin(omega * t)) * self.phi_k(xi, k)
                solution[i] = sol
            return solution
        else:
            solution = 0.0
            for k in range(1, n_terms + 1):
                Ak = self.compute_Ak(k)
                Bk = self.compute_Bk(k)
                omega = self.a * k * np.pi / self.l
                solution += (Ak * np.cos(omega * t) + Bk * np.sin(omega * t)) * self.phi_k(x, k)
            return solution


class WavePlotter:
    def __init__(self, wave_eq):
        self.wave_eq = wave_eq

    def plot_initial_conditions(self):
        x = np.linspace(0, self.wave_eq.l, 100)
        plt.figure(figsize=(10, 6))
        plt.plot(x, [self.wave_eq.u0(xi) for xi in x], label='u(x, 0)')
        plt.plot(x, [self.wave_eq.ut0(xi) for xi in x], label='∂u/∂t(x, 0)')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title('Начальные условия')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_analytical_solution(self, t_values, n_terms=10):
        x = np.linspace(0, self.wave_eq.l, 100)
        plt.figure(figsize=(10, 6))
        for t in t_values:
            u = self.wave_eq.analytical_solution(x, t, n_terms)
            plt.plot(x, u, label=f't = {t}')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title('Аналитическое решение')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_fourier_coefficients(self, n_terms=10):
        k = np.arange(1, n_terms + 1)
        Ak = [self.wave_eq.compute_Ak(ki) for ki in k]
        Bk = [self.wave_eq.compute_Bk(ki) for ki in k]

        plt.figure(figsize=(10, 6))
        plt.stem(k, Ak, linefmt='b-', markerfmt='bo', basefmt='b-', label='A_k')
        plt.stem(k, Bk, linefmt='r-', markerfmt='ro', basefmt='r-', label='B_k')
        plt.xlabel('k')
        plt.ylabel('Coefficients')
        plt.title('Коэффициенты Фурье')
        plt.legend()
        plt.grid(True)
        plt.show()


class CaseG(WaveEquation):
    def __init__(self, l):
        super().__init__(l)
        self.u0 = lambda x: 0.0
        self.ut0 = lambda x: np.sin(2 * np.pi * x / l)
        self.set_initial_conditions(self.u0, self.ut0)


class CaseD(WaveEquation):
    def __init__(self, l):
        super().__init__(l)
        self.u0 = lambda x: x ** 2
        self.ut0 = lambda x: 0.0
        self.set_initial_conditions(self.u0, self.ut0)


class CaseE(WaveEquation):
    def __init__(self, l):
        super().__init__(l)
        self.u0 = lambda x: x
        self.ut0 = lambda x: -x
        self.set_initial_conditions(self.u0, self.ut0)


# Пример использования
if __name__ == "__main__":
    l = np.pi / 2  # Длина интервала
    case_g = CaseG(l)

    case_g.analytical_solution()
    # plotter_g = WavePlotter(case_g)
    #
    # plotter_g.plot_initial_conditions()
    # plotter_g.plot_analytical_solution(t_values=[0.0, 0.5, 1.0], n_terms=5)
    # plotter_g.plot_fourier_coefficients(n_terms=5)