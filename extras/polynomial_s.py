import math
import numpy as np



def energy_expectation_polynomial_expansion_smin(
    dbi_object, d: np.array, n: int = 3, state=0
):
    """Return the first root of the Taylor expansion coefficients of energy expectation of `dbi_object` with respect to double bracket rotation duration `s`."""
    # generate Gamma's where $\Gamma_{k+1}=[W, \Gamma_{k}], $\Gamma_0=H
    Gamma_list = dbi_object.generate_gamma_list(n + 1, d)
    # coefficients
    coef = np.empty(n)
    state_cast = dbi_object.backend.cast(state)
    state_dag = dbi_object.backend.cast(state.conj().T)
    exp_list = np.array([1 / math.factorial(k) for k in range(n + 1)])
    for i in range(n):
        coef[i] = np.real(
            exp_list[i] * state_dag @ Gamma_list[i+1] @ state_cast
        )
    coef = list(reversed(coef))

    roots = np.roots(coef)
    real_positive_roots = [
        np.real(root) for root in roots if np.imag(root) < 1e-3 and np.real(root) > 0
    ]
    # solution exists, return minimum s
    if len(real_positive_roots) > 0:
        losses = [dbi_object.loss(step=root, d=d) for root in real_positive_roots]
    return real_positive_roots[losses.index(min(losses))]