# /// script
# requires-python = ">=3.12.2"
# dependencies = [
#     "ipykernel",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "jax",
#     "marimo",
# ]
# ///

# %%

# To sync the local venv with these requirements, run:
# #!VIRTUAL_ENV=.venv uv sync --script example.py --active
#!%reset -f
#!%load_ext autoreload
#!%autoreload now


import numpy as np
from collocation import solve

import matplotlib.pyplot as plt

def dzdt(t, z):
    return z**2 - 2*z + 1

def analytical(t):
    return (4*t - 3)/(4*t + 1)

t0 = 0
tf = 4
breakpoint()
solution = solve(dzdt, analytical(t0), t0, tf, K=4, representation_str='monomial', N=1)
# New problem now: Why does this do so poorly on [2, 4] for this problem? It should be
# so easy, it's practically a fucking line. And yet not even a linear model captures it. wtf.
# Update: It seems the initial guess was poor. I should base it on making sure that the
# first element of the initial guess is equal to x0, and for the next term I could evaluate
# the derivative at t0, x0 by using dxdt, and then I think I could keep taking the derivative
# wrt time until I get to 0 for the remaining constants, and then 0 for the rest.
# Might be overkill, and continued derivatives might get cumbersome, but at least the first
# and second terms can have reasonable guesses.

plt.plot(t:=np.linspace(t0, tf), analytical(t), color='b')
plt.plot(t, [solution(ti) for ti in t], color='r')
plt.grid()

error = solution(tf) - analytical(tf)
print("Error in last position: ", error)

from scipy.integrate import quad

error = quad(lambda t: np.abs(solution(t) - analytical(t)), 0, 1)
print("Total integrated error", error[0])
