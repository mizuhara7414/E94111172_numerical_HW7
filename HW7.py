import numpy as np



# Setting up the linear system Ax = b
A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 1, -1],
    [-1, 0, 0, 4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
])

b = np.array([0, -1, 9, 4, 8, 6])






# (a) Jacobi Method
def jacobi_method(A, b, max_iterations=100, tolerance=1e-10):
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    iterations = 0
    residuals = []
    
    while iterations < max_iterations:
        for i in range(n):
            sum_term = 0
            for j in range(n):
                if i != j:
                    sum_term += A[i, j] * x[j]
            x_new[i] = (b[i] - sum_term) / A[i, i]
        
        residual = np.linalg.norm(x_new - x)
        residuals.append(residual)
        
        if residual < tolerance:
            break
            
        x = x_new.copy()
        iterations += 1
    
    return x, iterations, residuals

# (b) Gauss-Seidel Method
def gauss_seidel_method(A, b, max_iterations=100, tolerance=1e-10):
    n = len(b)
    x = np.zeros(n)
    iterations = 0
    residuals = []
    
    while iterations < max_iterations:
        x_old = x.copy()
        for i in range(n):
            sum_term = 0
            for j in range(n):
                if i != j:
                    sum_term += A[i, j] * x[j]
            x[i] = (b[i] - sum_term) / A[i, i]
        
        residual = np.linalg.norm(x - x_old)
        residuals.append(residual)
        
        if residual < tolerance:
            break
            
        iterations += 1
    
    return x, iterations, residuals

# (c) SOR  Method
def sor_method(A, b, omega=1.5, max_iterations=100, tolerance=1e-10):
    n = len(b)
    x = np.zeros(n)
    iterations = 0
    residuals = []
    
    while iterations < max_iterations:
        x_old = x.copy()
        for i in range(n):
            sum_term = 0
            for j in range(i):
                sum_term += A[i, j] * x[j]
            for j in range(i+1, n):
                sum_term += A[i, j] * x_old[j]
            
            x[i] = (1 - omega) * x_old[i] + omega * (b[i] - sum_term) / A[i, i]
        
        residual = np.linalg.norm(x - x_old)
        residuals.append(residual)
        
        if residual < tolerance:
            break
            
        iterations += 1
    
    return x, iterations, residuals

# (d) Conjugate Gradient Method
def conjugate_gradient_method(A, b, max_iterations=100, tolerance=1e-10):
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)
    residuals = [np.sqrt(rs_old)]
    
    for i in range(max_iterations):
        Ap = A @ p
        denom = np.dot(p, Ap)
        if abs(denom) < 1e-14:
            break
        alpha = rs_old / denom
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        residual = np.sqrt(rs_new)
        residuals.append(residual)
        
        if residual < tolerance:
            return x, i + 1, residuals
        
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
        
    return x, max_iterations, residuals



print("(a) Jacobi Method:")
x_jacobi, iter_jacobi, res_jacobi = jacobi_method(A, b)
print(f"Solution after {iter_jacobi} iterations:")
print(x_jacobi)
print()

print("(b) Gauss-Seidel Method:")
x_gs, iter_gs, res_gs = gauss_seidel_method(A, b)
print(f"Solution after {iter_gs} iterations:")
print(x_gs)

print()

print("(c) SOR Method (ω=1.5):")
x_sor, iter_sor, res_sor = sor_method(A, b)
print(f"Solution after {iter_sor} iterations:")
print(x_sor)

print()

print("(d) Conjugate Gradient Method:")
x_cg, iter_cg, res_cg = conjugate_gradient_method(A, b)
print(f"Solution after {iter_cg} iterations:")
print(x_cg)

print()

