"""
Measurement function used by the DSSE.

After this revision the measurement vector **h(x)** is, for a system with
*NphaseNodes*=len(busphase_map),

    ┌            ┐
    │   P_inj    │   (size NphaseNodes)
    │   Q_inj    │   (size NphaseNodes)
    │   |V|      │   (size NphaseNodes)
    └            ┘
    length = 3 · NphaseNodes

Everything is expressed in per‑unit.  The ordering of the state vector
x = [Vm(0…N‑1), Va(0…N‑1)] (rad) is unchanged.
"""
from __future__ import annotations

import numpy as np

__all__ = ["measurement_function"]


# --------------------------------------------------------------------------
# Core measurement model
# --------------------------------------------------------------------------
def measurement_function(
    x: np.ndarray,
    Ybus: np.ndarray,
    mpc: dict,
    busphase_map: dict[int, int],
) -> np.ndarray:
    """
    Parameters
    ----------
    x
        State vector [Vm (p.u.), Va (rad)].
    Ybus
        Phase‑domain bus admittance matrix in per‑unit.
    mpc
        The MATPOWER‑style network dictionary (not used here but kept for
        forward compatibility).
    busphase_map
        (bus,phase) ➜ flat node index mapping – only its length matters.

    Returns
    -------
    h : np.ndarray
        Concatenated [P_inj, Q_inj, |V|] vector.
    """
    nnode = len(busphase_map)
    Vm, Va = x[:nnode], x[nnode:]

    # Complex voltages
    V = Vm * np.exp(1j * Va)

    # Nodal current injections (positive into the network)
    I = Ybus @ V

    # Complex power injections
    S = V * np.conj(I)
    P = S.real
    Q = S.imag

    # Assemble measurement vector
    h = np.concatenate([P, Q, Vm])

    return h
