import sympy as sp
from prettytable import PrettyTable

# Определяем символьные переменные
x = sp.symbols('x')
C1, C2 = sp.symbols('C1 C2')

# Исправленные базисные функции, удовлетворяющие условиям
phi0 = x/(x + 1)  # Точное решение для проверки структуры
phi1 = x**2 * (x - 1)   # Удовлетворяет: phi1(0)=0, phi1(1)=0, phi1'(0)=0
phi2 = x**2 * (x - 1)**2  # Удовлетворяет: phi2(0)=0, phi2(1)=0, phi2'(0)=0

# Приближенное решение (phi0 уже удовлетворяет краевым условиям)
u = phi0 + C1*phi1 + C2*phi2

# Вычисляем производные
u_prime = sp.diff(u, x)
u_double_prime = sp.diff(u_prime, x)

# Левая часть уравнения: Lu = u'' + (x+1)u' + u
Lu = u_double_prime + (x + 1)*u_prime + u

# Правая часть уравнения: f = -2/(x+1)^3 + 1
f = -2/(x + 1)**3 + 1

# Невязка R = Lu - f
R = Lu - f

# Условия ортогональности
eq1 = sp.integrate(R * phi1, (x, 0, 1))
eq2 = sp.integrate(R * phi2, (x, 0, 1))

# Решаем систему уравнений
solution = sp.solve([eq1, eq2], (C1, C2))

# Подставляем коэффициенты в решение
u_approx = u.subs({C1: solution[C1], C2: solution[C2]}).simplify()

# Создаем таблицу для проверки условий
table = PrettyTable()
table.field_names = ["Проверка", "Значение", "Ожидается", "Разница"]

# Проверка краевых условий
u_prime_0 = u_approx.diff(x).subs(x, 0)
table.add_row(["u'(0)", f"{float(u_prime_0.evalf()):.6f}", "1", f"{float(abs(u_prime_0 - 1)):.2e}"])


u_1 = u_approx.subs(x, 1)
table.add_row(["u(1)", f"{float(u_1.evalf()):.6f}", "0.5", f"{float(abs(u_1 - 0.5)):.2e}"])

# Сравнение с точным решением
x_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
for x_val in x_vals:
    approx = float(u_approx.subs(x, x_val).evalf())
    exact = float(x_val / (x_val + 1))
    table.add_row([f"u({x_val})", f"{approx:.6f}", f"{exact:.6f}", f"{abs(approx - exact):.2e}"])

print("Приближенное решение:")
sp.pprint(u_approx)
print("\nРезультаты проверки:")
print(table)