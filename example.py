# %%

# To sync the local venv with these requirements, run:
#!uv sync --all-extras
#!%reset_selective -f ^(?!(debugpy|get_ipython|os|trace_this_thread)$).*$
#!%load_ext autoreload
#!%autoreload now
#!%matplotlib widget


import numpy as np
import jax.numpy as jnp
from math import sin
from collocation import solve
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
plt.clf()

def dzdt(t, z):# return z**2 - 2*z + 1
    # return z[0]
    return [z[1], -z[0]]

def analytical(t):
    return (4*t - 3)/(4*t + 1)

t0 = 0
tf = 1
x0 = np.array([1, 0])
solution = solve(dzdt, x0, t0, tf, K=3, representation_str='monomial', N=1)

# x0 = np.atleast_1d(x0)
solivp = solve_ivp(dzdt, [t0, tf], x0, t_eval=np.linspace(t0, tf))


plt.plot(solivp.t, solivp.y[0], color='b', label="ground truth")
plt.plot(solivp.t, [solution(ti)[0] for ti in solivp.t], color='r', linestyle="--")
plt.grid()
plt.legend()
plt.show()

from scipy.integrate import quad

# error = quad(lambda t: np.abs(solution(t) - analytical(t)), 0, 1)
# print("Total integrated error", error[0])
