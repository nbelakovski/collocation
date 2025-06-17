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


import numpy as np
from collocation import solve, representation

import matplotlib.pyplot as plt

def dzdt(t, z):
    return z**2 - 2*z + 1

def analytical(t):
    return (4*t - 3)/(4*t + 1)

coefficients = solve(dzdt, -3, 0, 1, 5)

plt.plot(t:=np.linspace(0, 1), analytical(t), color='b')
plt.plot(t, [representation(ti, coefficients) for ti in t], color='r')
plt.grid()

error = representation(1, coefficients) - analytical(1)
print("Error in last position: ", error)

from scipy.integrate import quad

error = quad(lambda t: np.abs(representation(t, coefficients) - analytical(t)), 0, 1)
print("Total integrated error", error[0])
