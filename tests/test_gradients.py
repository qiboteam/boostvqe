"""Testing compute gradient function."""
import numpy as np
import pytest
from qibo import Circuit, gates, hamiltonians, set_backend

from .. import ansatze


@pytest.mark.parametrize("nqubits", [1, 2, 5, 10])
def test_compute_gradients(nqubits):
    """Test compute gradients"""
    set_backend("numpy")

    h = hamiltonians.Z(nqubits)
    c = Circuit(nqubits)
    c.add(gates.RX(i, np.pi) for i in range(nqubits))

    target_state = np.zeros((nqubits, 1), dtype=np.complex128)
    np.testing.assert_allclose(
        ansatze.compute_gradients(c.get_parameters(), c, h), target_state, atol=1e-7
    )
