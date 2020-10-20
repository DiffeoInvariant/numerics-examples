#!/usr/bin/env JAX_ENABLE_X64=True python3
import jax
import jax.numpy as np
from functools import partial
import matplotlib.pyplot as plt

@partial(jax.jit,static_argnums=(0,1))
def forward_euler(f,n,h,x0):
    xt = [x0]
    x = x0
    #x_{t+1} = x_t + h * f(x_t)
    for i in range(n):
        x = x + h * f(x)
        xt.append(x)

    return xt

def basic_newton_solve(f, x0, tol=1.0e-8, maxiter=1000):
    jac_fn = jax.jacobian(f)
    x = x0 + 1
    for i in range(maxiter):
        y = f(x0)
        J = jac_fn(x0)
        x = x0 - np.linalg.solve(J,y)
        deltax = np.linalg.norm(x - x0)
        if deltax < tol:
            return x
        x0 = x

    raise Exception(f"Error, maximum iterations {maxiter} exceeded and solution is not within tolerance {tol}! Current residual norm is {np.linalg.norm(y)}. Either increase tol or maxiter.")


def backward_euler(f,n,h,x0):
    xt = [x0]
    x = x0
    #x_{t+1} = x_t + h * f(x_{t+1}) -> x_{t+1} - h * f(x_{t+1}) - x_t = 0
    @jax.jit
    def _F(x_t, x_tp1):
        return x_tp1 - h * f(x_tp1) - x_t
    
    for i in range(n):
        x = basic_newton_solve(lambda val: _F(x0,val), x0, 1.0e-8, 1000)
        xt.append(x)
        x0 = x

    return xt


def crank_nicolson(f,n,h,x0):
    xt = [x0]
    x = x0
    # x_{t+1} - 0.5 * h * (f(x_{t+1}) + f(x_t)) - x_t = 0
    @jax.jit
    def _F(x_t, x_tp1):
        return x_tp1 - 0.5 * h * (f(x_tp1) + f(x_t)) - x_t

    for i in range(n):
        x = basic_newton_solve(lambda val: _F(x0,val), x0, 1.0e-8, 1000)
        xt.append(x)
        x0 = x

    return xt
    
@jax.jit
def rhs(x):
    return np.array([x[1],-x[0]])

def unwrap_solution(soln):
    x, y = [], []
    for val in soln:
        x.append(val[0])
        y.append(val[1])
    return x,y

x0 = np.array([1.0,0.0])
h = 0.1
n = 100
print(f"running forward euler")
fwd_soln = forward_euler(rhs,n,h,x0)
print(f"running backward euler. last point was {fwd_soln[-1]}")
bwd_soln = backward_euler(rhs,n,h,x0)
print(f"Running crank-nicolson")
cn_soln = crank_nicolson(rhs,n,h,x0)

fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)

px,py = unwrap_solution(fwd_soln)
ax1.plot(px,py)
ax1.set_title(f'Forward Euler Solution starting at {x0} ({n} timesteps with dt={h})')
px, py = unwrap_solution(bwd_soln)

ax2.plot(px,py)
ax2.set_title(f'Backward Euler solution starting at {x0}')

px,py = unwrap_solution(cn_soln)
ax3.plot(px,py)
ax3.set_title(f'Crank-Nicolson solution starting at {x0}')

plt.show()
