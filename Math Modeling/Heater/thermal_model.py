from scipy.integrate import solve_ivp

class ThermalSystem:
    def __init__(self, params):
        self.P = params['P']  # Мощность
        self.m = params['m']  # Масса
        self.c = params['c']  # Теплоёмкость
        self.S = params['S']  # Площадь поверхности
        self.k = params['k']  # Коэффициент теплоотдачи
        self.T0 = params['T0']  # Начальная температура
        self.sigma = 5.67e-8  # Постоянная Стефана-Больцмана
    def model_without_control(self, t, T):
        return (self.P
                - self.k * self.S * (T - self.T0)
                - self.sigma * self.S * (T ** 4 - self.T0 ** 4)) / (self.m * self.c)
    def model_with_control(self, t, T, T_max, T_min):
        self.H = 1 if (T < T_min) else (0 if (T > T_max) else self.H)
        return (self.P * self.H
                - self.k * self.S * (T - self.T0)
                - self.sigma * self.S * (T ** 4 - self.T0 ** 4)
                ) / (self.m * self.c)

    def solve(self, t_span, control=False, T_limits=None):
        if control:
            self.H = 1
            return solve_ivp(
                lambda t, T: self.model_with_control(t, T, *T_limits),
                t_span, [self.T0], max_step=1
            )
        else:
            return solve_ivp(
                self.model_without_control, t_span, [self.T0], max_step=1
            )
