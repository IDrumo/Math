import numpy as np
import matplotlib.pyplot as plt

def solve_heat_equation_splitting():
    # Parameters
    Lx, Ly = 1.0, 1.0
    Nx, Ny = 10, 10
    dx = Lx / Nx
    dy = Ly / Ny
    dt = 0.01
    T = 0.1
    Nt = int(T / dt)

    exact_solution_index = 4

    # Define exact solutions and source terms
    def exact_solution(x, y, t):
        if exact_solution_index == 1:
            return t * np.exp(x + y)
        elif exact_solution_index == 2:
            return t * np.sin(np.pi * x) * np.sin(np.pi * y)
        elif exact_solution_index == 3:
            return t + x**2 + y**2
        elif exact_solution_index == 4:
            return t + 0.25 * (x**2 + y**2)
        else:
            raise ValueError("Invalid exact solution index")

    def source_term(x, y, t):
        u = exact_solution(x, y, t)
        if exact_solution_index == 1:
            return np.exp(x + y) + 2 * t * np.exp(x + y)
        elif exact_solution_index == 2:
            return (np.sin(np.pi * x) * np.sin(np.pi * y) +
                    2 * np.pi**2 * t * np.sin(np.pi * x) * np.sin(np.pi * y))
        elif exact_solution_index == 3:
            return 1.0 + 2.0
        elif exact_solution_index == 4:
            return 1.0 + 0.5
        else:
            raise ValueError("Invalid exact solution index")

    # Grid
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    X, Y = np.meshgrid(x, y)

    # Initialize solution
    u = np.zeros((Ny+1, Nx+1))
    u_exact = np.zeros((Ny+1, Nx+1))

    # Initial condition (t=0)
    for i in range(Ny+1):
        for j in range(Nx+1):
            u[i, j] = exact_solution(X[i, j], Y[i, j], 0.0)

    # Precompute coefficients for Thomas algorithm
    alpha_x = dt / dx**2
    alpha_y = dt / dy**2

    # Thomas algorithm helper functions for tridiagonal systems
    def thomas(a, b, c, d):
        n = len(d)
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        for i in range(1, n):
            c_prime[i] = c[i] / (b[i] - a[i] * c_prime[i-1])
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / (b[i] - a[i] * c_prime[i-1])
        x = np.zeros(n)
        x[-1] = d_prime[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
        return x

    # Time stepping
    for n in range(Nt):
        t = (n + 1) * dt
        # X-direction solve (implicit)
        for i in range(Ny+1):
            # Dirichlet boundary conditions
            u_top = exact_solution(0.0, y[i], t)
            u_bot = exact_solution(Lx, y[i], t)
            # Construct tridiagonal matrix coefficients
            a = np.zeros(Nx+1)
            b = np.zeros(Nx+1)
            c = np.zeros(Nx+1)
            d = np.zeros(Nx+1)
            a[1:] = -alpha_x
            c[:-1] = -alpha_x
            b[:] = 1.0 + 2.0 * alpha_x
            b[0] = 1.0  # Boundary condition
            b[-1] = 1.0  # Boundary condition
            d[0] = u_top
            d[-1] = u_bot
            for j in range(1, Nx):
                d[j] = u[i, j] + alpha_x * (u[i, j-1] - 2*u[i, j] + u[i, j+1]) + dt * source_term(x[j], y[i], t)
            # Solve tridiagonal system
            u_mid = thomas(a, b, c, d)
            u[i, :] = u_mid

        # Y-direction solve (implicit)
        for j in range(Nx+1):
            # Dirichlet boundary conditions
            u_left = exact_solution(x[j], 0.0, t)
            u_right = exact_solution(x[j], Ly, t)
            # Construct tridiagonal matrix coefficients
            a = np.zeros(Ny+1)
            b = np.zeros(Ny+1)
            c = np.zeros(Ny+1)
            d = np.zeros(Ny+1)
            a[1:] = -alpha_y
            c[:-1] = -alpha_y
            b[:] = 1.0 + 2.0 * alpha_y
            b[0] = 1.0  # Boundary condition
            b[-1] = 1.0  # Boundary condition
            d[0] = u_left
            d[-1] = u_right
            for i in range(1, Ny):
                d[i] = u[i, j] + alpha_y * (u[i-1, j] - 2*u[i, j] + u[i+1, j]) + dt * source_term(x[j], y[i], t)
            # Solve tridiagonal system
            u_mid = thomas(a, b, c, d)
            u[:, j] = u_mid

        # Compute exact solution for error analysis
        for i in range(Ny+1):
            for j in range(Nx+1):
                u_exact[i, j] = exact_solution(X[i, j], Y[i, j], t)

        # Calculate error (L2 norm)
        error = np.sqrt(np.sum((u - u_exact)**2) / (Nx * Ny))
        print(f"Time step {n+1}, Error: {error:.6e}")

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, u, levels=20)
    plt.colorbar(label='Numerical Solution')
    plt.title('Numerical Solution')

    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, u_exact, levels=20)
    plt.colorbar(label='Exact Solution')
    plt.title('Exact Solution')
    plt.show()

# Run the solver
solve_heat_equation_splitting()