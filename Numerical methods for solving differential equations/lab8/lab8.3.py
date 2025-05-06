import numpy as np
import matplotlib.pyplot as plt

# Parameters
h = 0.1
N = 10  # Number of intervals
tau = 0.002  # Time step (chosen to ensure stability)
T = 0.1
M = int(T / tau)
sigma2 = 0.5 - h**2 / (12 * tau)

# Grid points
x = np.linspace(0, 1, N+1)
y = np.linspace(0, 1, N+1)

# Initialize solution and exact solution
u = np.zeros((N+1, N+1))
u_exact = np.zeros((N+1, N+1))

# Precompute Thomas algorithm coefficients
alpha_x = -0.5 * tau / h**2
beta_x = 1 + tau / h**2
alpha_y = -sigma2 * tau / h**2
beta_y = 1 + 2 * sigma2 * tau / h**2

# Thomas algorithm function
def thomas(a, b, c, d):
    n = len(d)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n-1):
        c_prime[i] = c[i] / (b[i] - a[i-1] * c_prime[i-1])
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / (b[i] - a[i-1] * c_prime[i-1])
    d_prime[-1] = (d[-1] - a[-1] * d_prime[-2]) / (b[-1] - a[-1] * c_prime[-2])
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    return x

# Time-stepping loop
for n in range(M):
    t = n * tau
    t_half = t + tau / 2
    t_new = t + tau

    # Compute forcing term phi^n = -e^{x + y}
    phi = -np.exp(x[:, np.newaxis] + y[np.newaxis, :])

    # First substep: solve for u_half in x-direction
    u_half = np.zeros_like(u)

    # Compute second derivative in y for the right-hand side
    laplacian_y = np.zeros_like(u)
    for i in range(N+1):
        for j in range(N+1):
            if j == 0:
                u_j_minus = t * np.exp(x[i] + y[j] - h)
                u_j_plus = u[i, j+1]
            elif j == N:
                u_j_plus = t * np.exp(x[i] + y[j] + h)
                u_j_minus = u[i, j-1]
            else:
                u_j_minus = u[i, j-1]
                u_j_plus = u[i, j+1]
            laplacian_y[i, j] = (u_j_plus - 2 * u[i, j] + u_j_minus) / h**2

    rhs = u + tau * (1 - sigma2) * laplacian_y + tau * phi

    # Solve for each row using Thomas algorithm
    for j in range(N+1):
        # Set boundary conditions for u_half at x=0 and x=1
        u_half[0, j] = t_half * np.exp(x[0] + y[j])
        u_half[N, j] = t_half * np.exp(x[N] + y[j])

        # Extract the interior part of the rhs
        d_interior = rhs[1:N, j].copy()
        if N > 1:
            d_interior[0] -= alpha_x * u_half[0, j]
            d_interior[-1] -= alpha_x * u_half[N, j]

        # Coefficients for Thomas algorithm
        a = np.full(N-2, alpha_x)
        b = np.full(N-1, beta_x)
        c = np.full(N-2, alpha_x)

        # Solve the tridiagonal system
        if N > 1:
            u_interior = thomas(a, b, c, d_interior)
            u_half[1:N, j] = u_interior

    # Second substep: solve for u_new in y-direction
    u_new = np.zeros_like(u_half)

    # Compute second derivative in x for the right-hand side
    laplacian_x = np.zeros_like(u_half)
    for j in range(N+1):
        for i in range(N+1):
            if i == 0:
                u_i_minus = t_half * np.exp(x[i] - h + y[j])
                u_i_plus = u_half[i+1, j]
            elif i == N:
                u_i_plus = t_half * np.exp(x[i] + h + y[j])
                u_i_minus = u_half[i-1, j]
            else:
                u_i_minus = u_half[i-1, j]
                u_i_plus = u_half[i+1, j]
            laplacian_x[i, j] = (u_i_plus - 2 * u_half[i, j] + u_i_minus) / h**2

    rhs = u_half + 0.5 * tau * laplacian_x + tau * phi

    # Solve for each column using Thomas algorithm
    for i in range(N+1):
        # Set boundary conditions for u_new at y=0 and y=1
        u_new[i, 0] = t_new * np.exp(x[i] + y[0])
        u_new[i, N] = t_new * np.exp(x[i] + y[N])

        # Extract the interior part of the rhs
        d_interior = rhs[i, 1:N].copy()
        if N > 1:
            d_interior[0] -= alpha_y * u_new[i, 0]
            d_interior[-1] -= alpha_y * u_new[i, N]

        # Coefficients for Thomas algorithm
        a = np.full(N-2, alpha_y)
        b = np.full(N-1, beta_y)
        c = np.full(N-2, alpha_y)

        # Solve the tridiagonal system
        if N > 1:
            u_interior = thomas(a, b, c, d_interior)
            u_new[i, 1:N] = u_interior

    # Update u and compute error
    u = u_new.copy()
    for i in range(N+1):
        for j in range(N+1):
            u_exact[i, j] = t_new * np.exp(x[i] + y[j])
    error = np.max(np.abs(u - u_exact))
    print(f"Time step {n+1}, error: {error}")

# Plotting results
X, Y = np.meshgrid(x, y, indexing='ij')

plt.figure()
plt.contourf(X, Y, u, 20)
plt.colorbar()
plt.title('Numerical Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure()
plt.contourf(X, Y, u_exact, 20)
plt.colorbar()
plt.title('Exact Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure()
plt.contourf(X, Y, np.abs(u - u_exact), 20)
plt.colorbar()
plt.title('Error')
plt.xlabel('x')
plt.ylabel('y')
plt.show()