import numpy as np
import jax.numpy as jnp
from scipy.special import roots_legendre
from scipy.optimize import root
from jax import grad, jacfwd, jacrev
from typing import Protocol


def product(a):
    result = 1
    for ai in a:
        result *= ai
    return result


def create_lagrange_representation(points):
    def lj(t, j):
        return jnp.prod(jnp.array([(t - points[m])/(points[j] - points[m]) for m in range(len(points)) if m != j]))
    def lagrange_representation(t, params):
        return jnp.dot(params, jnp.array([lj(t, j) for j in range(len(points))]))
    return lagrange_representation


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


def scaled_collocation_points(K, t0, tf):
    collocation_points, weights = roots_legendre(K)
    collocation_points += 1  # Shift from [-1, 1] to [0, 2]
    collocation_points *= (tf - t0)/2  # scale to [0, (tf-t0)]
    collocation_points += t0  # then shift to [t0, tf]
    return collocation_points


class Dynamics(Protocol):
    def __call__(self, t: float, z: list[float] | float) -> list[float] | float:
        ...

class Representation(Protocol):
    def __call__(self, t: float, params: list[float]) -> list[float] | float:
        ...

# Hm, this seems hardcoded to a single variable, what if x is a vector?
def solve(dxdt: Dynamics, x0, t0, tf, K, representation_str: str, N=2):  # dxdt - > function of t, x, representation
    # TODO: I'd like to try splitting up the domain with multiple elements, i.e. multiple
    # representations, and of course that means more coefficients.
    if N == 1 and representation_str == 'monomial':
        representation = monomial_representation
        collocation_points = scaled_collocation_points(K, t0, tf)
        colloc_matrix = jnp.ones((K+1, K+1))
        colloc_matrix = colloc_matrix.at[:, 0].set([t0**i for i in range(K+1)])
        for i in range(K+1):
            colloc_matrix = colloc_matrix.at[i, 1:].set(collocation_points**i)
        print(collocation_points)

        print("Colloc matrix", colloc_matrix)
        # roots = np.linspace(t0+(tf-t0)/K, tf, K-1)
        numvars = len(x0)
        ig = np.ones((K+1, numvars))  # initial guess
        ig[0, :] = x0  # This works for scalars but not systems of ODEs.
        print("aasdf", dxdt(t0, x0), "|", ig[1, :])
        ig[1, :] = dxdt(t0, x0)
        print("qwer", numvars)
        
        def create_system_of_eqns_monmial_1(t0, x0, dxdt, collocation_points, representation):
            reprgrad = grad(representation)
            def system(params):
                result = jnp.zeros(numvars*(K+1))
                params = params.reshape((K+1, numvars))
                xtilde = params.T@colloc_matrix  # (numvars, K+1)
                result = result.at[:numvars].set(xtilde[:, 0] - x0)
                for i in range(len(collocation_points)):
                    colloc1 = collocation_points[i]
                    dxdt_1 = jnp.array(dxdt(colloc1, xtilde[:, i+1]))  # This is of length N
                    reprgrad_1 = jnp.array([reprgrad(colloc1, params[:, j]) for j in range(numvars)])
                    result = result.at[numvars*(i+1):numvars*(i+2)].set(reprgrad_1 - dxdt_1)
                return result
            return system
        
        
        functosolve = create_system_of_eqns_monmial_1(t0, x0, dxdt, collocation_points, representation)
        functosolve(ig)
        sol = root(functosolve, x0=ig, jac=jacfwd(functosolve), method='hybr', tol=1e-3)
        print(sol)
        assert sol.success, "sol.success is false, the solution did not converge"
        sol.x = sol.x.reshape((K+1, numvars))
        return lambda t: [representation(t, sol.x[:, i]) for i in range(numvars)]
    elif N == 1 and representation_str == "lagrange":
        collocation_points = scaled_collocation_points(K, t0, tf)
        print(collocation_points)
        representation = create_lagrange_representation(np.hstack((t0, collocation_points)))

        ig = np.zeros(K+1)  # initial guess
        ig[0] = x0  # This works for scalars but not systems of ODEs.
        ig[1] = dxdt(t0, x0)
        functosolve = create_system_of_eqns(t0, x0, dxdt, collocation_points, representation)
        sol = root(functosolve, x0=ig, jac=jacfwd(functosolve), method='hybr', tol=1e-3)
        assert sol.success, "sol.success is false, the solution did not converge"
        return lambda t: representation(t, sol.x)
    elif N == 2 and representation_str == "monomial":
        representation = monomial_representation
        time_grid = np.linspace(t0, tf, N+1)
        element_1 = (time_grid[0], time_grid[1])
        element_2 = (time_grid[1], time_grid[2])
        collocation_points_1 = scaled_collocation_points(K, element_1[0], element_1[1])
        collocation_points_2 = scaled_collocation_points(K, element_2[0], element_2[1])

        def create_system_of_eqns2(t0, x0, dxdt, collocation_points, representation):
            reprgrad = grad(representation)
            def system(params):
                params1, params2 = (params[:K+1], params[K+1:])
                eq1 = [representation(t0, params1) - x0]
                eq21_to_n1 = [reprgrad(ti, params1) - dxdt(ti, representation(ti, params1)) for ti in collocation_points_1]
                eq_n1_to_22 = [representation(element_1[1], params1) - representation(element_2[0], params2)]
                eq22_to_n2 = [reprgrad(ti, params2) - dxdt(ti, representation(ti, params2)) for ti in collocation_points_2]
                return eq1 + eq21_to_n1 + eq_n1_to_22 + eq22_to_n2
            return system
        
        functosolve = create_system_of_eqns2(t0, x0, dxdt, None, representation)
        ig = np.ones(2*K+2)  # initial guess
        ig[0] = x0  # This works for scalars but not systems of ODEs.
        ig[1] = dxdt(t0, x0)
        ig[K+1] = x0 + ig[1] * (element_2[0] - element_1[0])
        sol = root(functosolve, x0=ig, jac=jacrev(functosolve), method='hybr', tol=1e-3)
        assert sol.success, "sol.success is false, the solution did not converge"
        # TODO: Obviously I will need a more complex solution for general N elements
        return lambda t: representation(t, sol.x[:K+1] if t < element_1[1] else sol.x[K+1:])

    elif N == 2 and representation_str == "lagrange":
        time_grid = np.linspace(t0, tf, N+1)
        element_1 = (time_grid[0], time_grid[1])
        element_2 = (time_grid[1], time_grid[2])
        collocation_points_1 = scaled_collocation_points(K, element_1[0], element_1[1])
        representation_1 = create_lagrange_representation(np.hstack((t0, collocation_points_1)))
        reprgrad_1 = grad(representation_1)
        collocation_points_2 = scaled_collocation_points(K, element_2[0], element_2[1])
        representation_2 = create_lagrange_representation(np.hstack((element_2[0], collocation_points_2)))
        reprgrad_2 = grad(representation_2)

        def create_system_of_eqns2(t0, x0, dxdt):
            def system(params):
                params1, params2 = (params[:K+1], params[K+1:])
                eq1 = [representation_1(t0, params1) - x0]
                eq21_to_n1 = [reprgrad_1(ti, params1) - dxdt(ti, representation_1(ti, params1)) for ti in collocation_points_1]
                eq_n1_to_22 = [representation_1(element_1[1], params1) - representation_2(element_2[0], params2)]
                eq22_to_n2 = [reprgrad_2(ti, params2) - dxdt(ti, representation_2(ti, params2)) for ti in collocation_points_2]
                return eq1 + eq21_to_n1 + eq_n1_to_22 + eq22_to_n2
            return system
        
        functosolve = create_system_of_eqns2(t0, x0, dxdt)
        ig = np.ones(N*(K+1))  # initial guess
        ig[0] = x0  # This works for scalars but not systems of ODEs.
        ig[1] = dxdt(t0, x0)
        # ig[K+1] = x0 + ig[1] * (element_2[0] - element_1[0])
        sol = root(functosolve, x0=ig, jac=jacrev(functosolve), method='hybr', tol=1e-3)
        assert sol.success, "sol.success is false, the solution did not converge"
        # TODO: Obviously I will need a more complex solution for general N elements
        return lambda t: representation_1(t, sol.x[:K+1]) if t < element_1[1] else representation_2(t, sol.x[K+1:])
    elif N > 2 and representation_str == "lagrange":

        time_grid = np.linspace(t0, tf, N+1)
        print(f'{time_grid=}')
        elements = list(zip(time_grid[:-1], time_grid[1:]))
        pts_repr_grad = []
        for element in elements:
            collocation_points = scaled_collocation_points(K, element[0], element[1])
            representation = create_lagrange_representation(np.hstack((element[0], collocation_points)))
            reprgrad = grad(representation)
            pts_repr_grad.append([collocation_points, representation, reprgrad])
        def create_system_of_eqns2(dxdt, x0, t0, tf, N):

            def system(params):
                pts, repres, reprgrad = pts_repr_grad[0]
                params1 = params[:K+1]
                eq1 = [repres(t0, params[:K+1]) - x0]
                eq21_to_n1 = [reprgrad(ti, params1) - dxdt(ti, repres(ti, params1)) for ti in pts]
                eqs = eq1 + eq21_to_n1
                for i in range(1, N):
                    params1 = params[(K+1)*(i-1):(K+1)*i]
                    params2 = params[(K+1)*i:(K+1)*(i+1)]
                    pts1, repres1, reprgrad1 = pts_repr_grad[i-1]
                    pts2, repres2, reprgrad2 = pts_repr_grad[i]
                    eq_n1_to_22 = [repres1(elements[i-1][1], params1) - repres2(elements[i][0], params2)]
                    eq22_to_n2 = [reprgrad2(ti, params2) - dxdt(ti, repres2(ti, params2)) for ti in pts2]
                    eqs += eq_n1_to_22
                    eqs += eq22_to_n2
                return eqs
            return system
        
        functosolve = create_system_of_eqns2(dxdt, x0, t0, tf, N)
        ig = -np.ones(N*(K+1))  # initial guess
        ig[0] = x0  # This works for scalars but not systems of ODEs.
        sol = root(functosolve, x0=ig, jac=jacrev(functosolve), method='hybr', tol=1e-3)
        assert sol.success, "sol.success is false, the solution did not converge"
        def final_representation(t):
            if t <= elements[0][0]:
                index = 0
            elif t >= elements[-1][1]:
                index = N-1
            else:
                for index, element in enumerate(elements):
                    if element[0] <= t <= element[1]:
                        break
            params = sol.x[(K+1)*index:(K+1)*(index+1)]
            return pts_repr_grad[index][1](t, params)
        return final_representation
    
    elif N > 2 and representation_str == "monomial":
        representation = monomial_representation

        time_grid = np.linspace(t0, tf, N+1)
        elements = list(zip(time_grid[:-1], time_grid[1:]))
        print(elements)
        def create_system_of_eqns2(dxdt, x0, t0, tf, N):
            reprgrad = grad(representation)
            def system(params):
                # We set up the equations for the first element manually, because that's
                # where we have to bring in the initial condition
                # pts = scaled_collocation_points(K, elements[0][0], elements[0][1])
                pts = np.linspace(elements[0][0], elements[0][1], K)
                params1 = params[:K+1]
                eq1 = [representation(t0, params[:K+1]) - x0]
                eq21_to_n1 = [reprgrad(ti, params1) - dxdt(ti, representation(ti, params1)) for ti in pts]
                eqs = eq1 + eq21_to_n1
                # For the rest of the elemetns we can do it in a loop
                for i in range(1, N):
                    params1 = params[(K+1) * (i-1) : (K+1) * i]
                    params2 = params[(K+1) * i     : (K+1) * (i+1)]
                    # Set the endpoint of element i-1 to be equal to the start of element i
                    eq_n1_to_22 = [representation(elements[i-1][1], params1) - representation(elements[i][0], params2)]
                    # Then set up the equations for element i
                    pts = scaled_collocation_points(K, elements[i][0], elements[i][1])
                    eq22_to_n2 = [reprgrad(ti, params2) - dxdt(ti, representation(ti, params2)) for ti in pts]
                    eqs += eq_n1_to_22
                    eqs += eq22_to_n2
                return eqs
            return system
        
        functosolve = create_system_of_eqns2(dxdt, x0, t0, tf, N)
        ig = np.zeros(N*(K+1))  # initial guess
        ig[0] = x0  # This works for scalars but not systems of ODEs.
        ig[1] = dxdt(t0, x0)
        sol = root(functosolve, x0=ig, jac=jacrev(functosolve), method='hybr', tol=1e-3)
        assert sol.success, "sol.success is false, the solution did not converge"
        print("Coefficients: ", ", ".join([str(e) for e in sol.x]))
        def final_representation(t):
            if t <= elements[0][0]:
                index = 0
            elif t >= elements[-1][1]:
                index = N-1
            else:
                for index, element in enumerate(elements):
                    if element[0] <= t <= element[1]:
                        break
            params = sol.x[(K+1)*index:(K+1)*(index+1)]
            return representation(t, params)
        return final_representation
