# /// script
# requires-python = ">=3.12.2"
# dependencies = [
#     "ipykernel",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "jax",
# ]
# ///

# %%

# To sync the local venv with these requirements, run:
#!VIRTUAL_ENV=.venv uv sync --script example.py --active
#!%reset -f
#!%load_ext autoreload
#!%autoreload now


import numpy as np
from collocation import solve, monomial_representation

import matplotlib.pyplot as plt

def dzdt(t, z):
    return z**2 - 2*z + 1

def analytical(t):
    return (4*t - 3)/(4*t + 1)

t0 = -0.1
tf = 4
repres = monomial_representation
coefficients = solve(dzdt, analytical(t0), t0, tf, K=9, representation=repres)
print(f'{coefficients=}')

plt.plot(t:=np.linspace(t0, tf), analytical(t), color='b')
plt.plot(t, [repres(ti, coefficients) for ti in t], color='r')
plt.grid()

error = repres(1, coefficients) - analytical(1)
print("Error in last position: ", error)

from scipy.integrate import quad

error = quad(lambda t: np.abs(repres(t, coefficients) - analytical(t)), 0, 1)
print("Total integrated error", error[0])
