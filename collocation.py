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
#!VIRTUAL_ENV=.venv uv sync --script collocation.py --active
#!%reset -f

# %% Here we start the generic code

import numpy as np
import jax.numpy as jnp
from scipy.special import roots_legendre
from scipy.optimize import root
from jax import grad, jacfwd, jacrev

def monomial_representation(t, params):
    monomial_basis = sum([params[i]*t**i for i in range(len(params))])
    return monomial_basis

def create_system_of_eqns(t0, x0, dxdt, collocation_points, representation):
    reprgrad = grad(representation)
    def system(params):
        eq1 = [representation(t0, params) - x0]
        eq2_to_n = [reprgrad(ti, params) - dxdt(ti, representation(ti, params)) for ti in collocation_points]
        return eq1 + eq2_to_n
    return system

# Hm, this seems hardcoded to a single variable, what if x is a vector?
def solve(dxdt, x0, t0, tf, K, representation, N=2):  # dxdt - > function of t, x
    # TODO: I'd like to try splitting up the domain with multiple elements, i.e. multiple
    # representations, and of course that means more coefficients.
    collocation_points, weights = roots_legendre(K-1)
    collocation_points += 1  # Shift from [-1, 1] to [0, 2]
    collocation_points *= (tf - t0)/2  # scale to [0, (tf-t0)]
    collocation_points += t0  # then shift to [t0, tf]
    # roots = np.linspace(t0+(tf-t0)/K, tf, K-1)
    
    functosolve = create_system_of_eqns(t0, x0, dxdt, collocation_points, representation)
    sol = root(functosolve, x0=np.ones(K), jac=jacrev(functosolve), method='hybr', tol=1e-3)  # np.linspace(t0, tf, K)
    assert sol.success, "sol.success is false, the solution did not converge"
    return sol.x
