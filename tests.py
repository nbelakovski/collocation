import numpy as np
from collocation import solve


def test_multi_element_state():
    def dzdt(t, z):
        return [z[1], -z[0]]

    t0 = 0
    tf = 1
    x0 = np.array([1, 0])
    # With this x0 the analytical solution is x(t) = cos(t)
    solution = solve(dzdt, x0, t0, tf, K=3, representation_str='monomial', N=1)
    assert np.allclose(solution(t0), x0)
    assert np.allclose(solution(tf), [np.cos(tf), -np.sin(tf)], rtol=1e-4, atol=1e-4)


def test_single_element_state():
    def dzdt(t, z):
        return [z[0]]

    t0 = 0
    tf = 1
    x0 = np.array([1])
    # With this x0 the analytical solution is x(t) = cos(t)
    solution = solve(dzdt, x0, t0, tf, K=3, representation_str='monomial', N=1)
    assert np.allclose(solution(t0), x0)
    assert np.allclose(solution(tf), [np.exp(tf)], rtol=1e-4, atol=1e-4)
