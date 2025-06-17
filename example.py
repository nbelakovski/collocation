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
from collocation import solve, representation

import matplotlib.pyplot as plt

def dzdt(t, z):
    return z**2 - 2*z + 1

def analytical(t):
    return (4*t - 3)/(4*t + 1)

t0 = -0.1
tf = 4
coefficients = solve(dzdt, analytical(t0), t0, tf, 9)
# Uniform grid failed to converge on 5, 6*, 8*, 9, 10
# Legendre roots failed to converge on 7, 10
# * It converged, but to an incorrect local minimum
# On 7, uniform grid had total error of 0.019473126645063787
# On 6, Legendre had error of 0.009999763241381451, ~50% better than uniform
# On 8 it got 0.001374732661305742 but there was an integration warning
# So choice of grid has a significant impact on convergence as well as on error
print(f'{coefficients=}')

plt.plot(t:=np.linspace(t0, tf), analytical(t), color='b')
plt.plot(t, [representation(ti, coefficients) for ti in t], color='r')
plt.grid()

error = representation(1, coefficients) - analytical(1)
print("Error in last position: ", error)

from scipy.integrate import quad

error = quad(lambda t: np.abs(representation(t, coefficients) - analytical(t)), 0, 1)
print("Total integrated error", error[0])
