# /// script
# requires-python = ">=3.12.2"
# dependencies = [
#     "ipykernel",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "jax",
#     "marimo",
#     "ipywidgets",
#     "ipympl"
# ]
# ///

# %%

# To sync the local venv with these requirements, run:
#!VIRTUAL_ENV=.venv uv sync --script example.py --active
#!%reset -f
#!%load_ext autoreload
#!%autoreload now
#!%matplotlib widget


import numpy as np
from collocation import solve

import matplotlib.pyplot as plt
plt.clf()

def dzdt(t, z):
    return z**2 - 2*z + 1

def analytical(t):
    return (4*t - 3)/(4*t + 1)

t0 = 0
tf = 4
solution = solve(dzdt, analytical(t0), t0, tf, K=4, representation_str='monomial', N=5)

plt.plot(t:=np.linspace(t0, tf), analytical(t), color='b')
plt.plot(t, [solution(ti) for ti in t], color='r')
plt.grid()
plt.show()

from scipy.integrate import quad

error = quad(lambda t: np.abs(solution(t) - analytical(t)), 0, 1)
print("Total integrated error", error[0])
