import numpy as np

# f(x, y) = x**2 + 2*(y**2) - 4*x - 8*y + 10
def grad(u):
    x, y = u
    return np.array([
        2*x - 4,
        4*y - 8
    ])

def hessian(u):
    x, y = u
    return np.array([
        [
            2, 0
        ],
        [
            0, 4
        ]
    ])

def newton_optim(u_0, grad_fn, hessian_fn, epsilon = 1e-8, max_iter=100):
    u = u_0
    for _ in range(max_iter):
        grad = grad_fn(u)
        if np.linalg.norm(grad) < epsilon:
            return u
        hessian = hessian_fn(u)
        t = np.linalg.solve(hessian, -grad)
        u = u + t
    return u

if __name__ == "__main__":
    u_0 = (0, 0)
    u_optim = newton_optim(u_0, grad, hessian)
    print(u_optim)