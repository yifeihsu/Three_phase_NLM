"""
Analytical Jacobian of the *reduced* measurement vector
h(x) = [P_inj, Q_inj, |V|]  (length = 3·N)

All notation follows the original code; only the branch‑flow part has
been removed and the row counters simplified.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps

__all__ = ["jacobian"]


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _dsbus_polar(Ybus: np.ndarray, V: np.ndarray, Vm: np.ndarray):
    """
    Returns the analytical partial derivatives of the complex power
    injection w.r.t. polar state variables:

        dSbus/dVa  and  dSbus/dVm
    """
    Ibus = Ybus @ V
    diagV = np.diag(V)
    diagI = np.diag(Ibus)
    Vnorm = V / Vm

    # dS/dVa
    term = diagI - Ybus @ np.diag(V)
    dSbus_dVa = 1j * diagV @ np.conjugate(term)

    # dS/dVm
    tmp1 = diagV @ np.conjugate(Ybus @ np.diag(Vnorm))
    tmp2 = np.conjugate(diagI) @ np.diag(Vnorm)
    dSbus_dVm = tmp1 + tmp2

    return dSbus_dVa, dSbus_dVm


# ---------------------------------------------------------------------------
# public routine
# ---------------------------------------------------------------------------
def jacobian(
    x: np.ndarray,
    Ybus: np.ndarray,
    mpc: dict,
    busphase_map: dict,
):
    """
    Parameters
    ----------
    x
        State vector: [Vm (p.u.), Va (rad)].
    Ybus
        Phase‑domain bus admittance.
    mpc, busphase_map
        Passed unchanged for interface compatibility.

    Returns
    -------
    H : scipy.sparse.csr_matrix
        Jacobian of size (3·N, 2·N).
    """
    nnode = len(busphase_map)
    half = nnode                      # index where angles start

    Vm = x[:half]
    Va = x[half:]
    V = Vm * np.exp(1j * Va)

    # Injection derivatives
    dS_dVa, dS_dVm = _dsbus_polar(Ybus, V, Vm)
    dS_dVm_r = dS_dVm.real
    dS_dVm_i = dS_dVm.imag
    dS_dVa_r = dS_dVa.real
    dS_dVa_i = dS_dVa.imag

    # Measurement counts
    m_inj = 2 * nnode                # P + Q
    m_v   = nnode                    # |V|
    m_tot = m_inj + m_v

    # Allocate sparse matrix
    H = sps.lil_matrix((m_tot, 2 * nnode))

    # -----------------------------------------------------------------------
    # 1) P_inj rows ---------------------------------------------------------
    H[0:nnode,          0:half] = dS_dVm_r
    H[0:nnode,          half:]  = dS_dVa_r

    # 2) Q_inj rows ---------------------------------------------------------
    H[nnode:2 * nnode,  0:half] = dS_dVm_i
    H[nnode:2 * nnode,  half:]  = dS_dVa_i

    # 3) |V| rows -----------------------------------------------------------
    offset_v = 2 * nnode
    for i in range(nnode):
        H[offset_v + i, i] = 1.0     # ∂|V|/∂Vm = 1

    return H.tocsr()
