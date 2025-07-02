"""
Analytical Jacobian of the reduced measurement vector with respect to
line parameters (12 per three‑phase line: 6 × R, 6 × X).

Only the 2·N bus‑injection rows are non‑zero; the |V| rows are identically
zero because voltage magnitude does not depend directly on line parameters.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

__all__ = ["jacobian_line_params"]


# ---------------------------------------------------------------------------
# helpers shared with the original code base
# ---------------------------------------------------------------------------
def build_param_info(mpc):
    """
    Returns an array of shape (n_params, 3):
        [ line_id, parameter_name, parameter_value ]
    keeping exactly the same ordering as the legacy implementation
    so that downstream indexing (Mahalanobis test, etc.) is unaffected.
    """
    line3p = mpc["line3p"]
    lc     = mpc["lc"]

    # lcid ➜ primitive matrix entries
    lc_dict = {int(row[0]): row[1:13] for row in lc}

    param_names = [
        "Zaa_R", "Zab_R", "Zac_R", "Zbb_R", "Zbc_R", "Zcc_R",
        "Zaa_X", "Zab_X", "Zac_X", "Zbb_X", "Zbc_X", "Zcc_X",
    ]

    info = []
    for line_row in line3p:
        line_id  = int(line_row[-1])
        lcid     = int(line_row[4])
        length_mi = line_row[5] / 5280.0

        if lcid == 0 or lcid not in lc_dict:
            continue

        base_vals = lc_dict[lcid] * length_mi  # per‑phase R/X over full length
        for j, pname in enumerate(param_names):
            info.append((line_id, pname, base_vals[j]))

    return np.asarray(info, dtype=object)


def build_dYsub_symbolic(Ysub, param_type):
    """
    Same utility as in the legacy code – generates ∂Y/∂p for the
    6×6 ‘from‑to’ sub‑matrix of the line under study.
    """
    pmap = {
        "Zaa_R": (0, 0, 1.0), "Zaa_X": (0, 0, 1j),
        "Zab_R": (0, 1, 1.0), "Zab_X": (0, 1, 1j),
        "Zac_R": (0, 2, 1.0), "Zac_X": (0, 2, 1j),
        "Zbb_R": (1, 1, 1.0), "Zbb_X": (1, 1, 1j),
        "Zbc_R": (1, 2, 1.0), "Zbc_X": (1, 2, 1j),
        "Zcc_R": (2, 2, 1.0), "Zcc_X": (2, 2, 1j),
    }

    dZ3 = np.zeros((3, 3), dtype=complex)
    if param_type in pmap:
        i, j, val = pmap[param_type]
        dZ3[i, j] = val
        if i != j:
            dZ3[j, i] = val  # symmetric entry

    # ∂Y = -Y · ∂Z · Y  (Π‑model linearisation)
    Y3   = -Ysub[3:, 0:3]
    dY3  = -Y3 @ (dZ3 @ Y3)

    dY6 = np.zeros((6, 6), dtype=complex)
    dY6[0:3, 0:3] = dY3
    dY6[3:, 3:]   = dY3
    dY6[0:3, 3:]  = -dY3
    dY6[3:, 0:3]  = -dY3
    return dY6


# ---------------------------------------------------------------------------
#  ∂h/∂p for a single parameter (injection rows only)
# ---------------------------------------------------------------------------
def _compute_param_injection_partial(
    x, Ybus, mpc, busphase_map, line_id, param_name
):
    nnode = len(busphase_map)
    m_tot = 3 * nnode                      # new measurement length
    dh    = np.zeros(m_tot, dtype=float)   # will hold ∂h/∂p

    half = nnode
    Vm = x[:half]
    Va = x[half:]
    V  = Vm * np.exp(1j * Va)

    # indices of the six phase nodes affected by this line
    line_data = mpc["line3p"][line_id - 1]
    fbus = int(line_data[1]) - 1
    tbus = int(line_data[2]) - 1
    f_idx = [fbus * 3 + k for k in range(3)]
    t_idx = [tbus * 3 + k for k in range(3)]
    rowcol = f_idx + t_idx                                # local ➜ global

    # 6×6 sub‑matrix of Ybus for the line
    Ysub = Ybus[np.ix_(rowcol, rowcol)]
    dY6  = build_dYsub_symbolic(Ysub, param_name)

    # ∂S_inj/∂p for each of the six involved node‑phases
    for local_i, g_idx in enumerate(rowcol):
        dI_param = np.sum(dY6[local_i, :] * V[rowcol])
        dSf      = V[g_idx] * np.conjugate(dI_param)
        dh[g_idx]             = dSf.real          # P row
        dh[g_idx + nnode]     = dSf.imag          # Q row
        # |V| rows (offset 2·N) stay zero

    return dh


# ---------------------------------------------------------------------------
#  public routine
# ---------------------------------------------------------------------------
def jacobian_line_params(x, Ybus, mpc, busphase_map):
    """
    Sparse Jacobian of size (3·N, n_params) where n_params = 12·n_lines
    (minus any lines skipped by build_param_info).
    """
    nnode = len(busphase_map)
    m_tot = 3 * nnode

    param_info = build_param_info(mpc)
    n_params   = len(param_info)

    data, rows, cols = [], [], []

    for c, (line_id, pname, _) in enumerate(param_info):
        dh_dp = _compute_param_injection_partial(
            x, Ybus, mpc, busphase_map, line_id, pname
        )
        nz = np.nonzero(dh_dp)[0]
        data.extend(dh_dp[nz])
        rows.extend(nz)
        cols.extend([c] * len(nz))

    return csr_matrix((data, (rows, cols)), shape=(m_tot, n_params))
