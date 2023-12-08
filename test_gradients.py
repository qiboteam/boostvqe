"""Testing compute gradient function."""
from ansatze import compute_gradients
import pytest
import numpy as np
from qibo import hamiltonians, set_backend, Circuit, gates


@pytest.mark.parametrize("nqubits", [1,2,5,10])
def test_compute_gradients(nqubits):
    """Test compute gradients"""
    set_backend("numpy")

    h = hamiltonians.Z(nqubits)
    c = Circuit(nqubits)
    c.add(gates.RX(i, np.pi) for i in range(nqubits))

    target_state = np.zeros((nqubits,1),dtype=np.complex128)
    np.testing.assert_allclose(compute_gradients(c.get_parameters(), c, h),
                               target_state,
                               atol=1e-7)
