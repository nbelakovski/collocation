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
from scipy.optimize import fsolve
from jax import grad, jacfwd, jacrev

def representation(t, params):
    monomial_basis = sum([params[i]*t**i for i in range(len(params))])
    return monomial_basis

reprgrad = grad(representation)

# Hm, this seems hardcoded to a single variable, what if x is a vector?
def solve(dxdt, x0, t0, tf, K, N=1):  # dxdt - > function of t, x
    roots, weights = roots_legendre(K)  # Gives K roots
    roots += 1
    roots /= 2
    roots = np.linspace(t0+(tf-t0)/K, tf, K)
    print(roots)
    # What I really need is to convert whatever representation I have into a matrix form
    # So that I can have Ax=b where x = [a0, a1, a2, ...] and b = dxdt(ti), and A is [1, t, t**2, ...; 1, t, t**2, ...; ...] where each row of A corresponds to a root from roots_legendre, except the first row which is the IC
    # I guess b is more of a vector of functions, since this is something I need to call repeatedly until I get the right values
    # Ax=b doesn't work if b is unknown.
    # I guess that's why we need an NLP? But what's my obj function in this case?
    # And if I used langrage interpolating polynomials I still have the problem that b is unknown, by the way
    # Or perhaps I can use fsolve, which finds the roots of func(x) = 0 given a starting point.
    # I can certainly phrase the problem as Ax-b = 0, and my unknowns are simply my coefficients
    # and the state at all the collocation points. But wait this fails, because for every equation I add,
    # I need one more. At least that's true with a monomial basis, it can work with a Lagrange basis.
    # Boyd substitutes the representation on the right side of the DE as well as on the left, which leads
    # things to simplify down to just the unknown coefficients, but this would require some sort of symbolic
    # logic, or at least it would if we wanted to reduce things, but I guess we could jsut use a nonlinear
    # solver and treat the right side as a black box that's just a function of t, i.e. our x is just coming
    # from our representation.
    def functosolve(x, t):
        eq1 = [representation(t[0], params=x) - x0]
        return eq1 + [reprgrad(ti, params=x) - dxdt(t, representation(ti, params=x)) for ti in t[1:]]
    result, infodict, ier, mesg = fsolve(functosolve, x0=np.ones(K), fprime=jacrev(functosolve), args=roots, full_output=True)  # np.linspace(t0, tf, K)
    print(mesg, functosolve(result, np.hstack((0, roots))))
    assert ier == 1, "ier !=1, the solution did not converge"

    # So, trying collocation with monomial basis of degree 4 and uniform grid got me a total error of 0.177. Degree 5 and above did not converge
    # Collocation with monomial basis and legendre roots only converged at K=2, i.e. linear
    return result
