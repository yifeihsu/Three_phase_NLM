import numpy as np
from scipy.sparse import csr_matrix

def jacobian_shunt_params(x, mpc, busphase_map):
    nnode = len(busphase_map)
    m_tot = 3 * nnode                 # [P,Q,|V|] for every node‑phase
    n_par = 2 * nnode                 #  G_k  and  B_k

    Vm = x[:nnode]                    # per‑unit magnitudes
    V2 = Vm ** 2                      # |V_k|² used in every column

    data, rows, cols = [], [], []

    for k in range(nnode):
        # column order: [G1,B1, G2,B2, …]
        col_G = 2 * k
        col_B = 2 * k + 1

        # rows for this node‑phase
        rP = k                # P injection
        rQ = k + nnode        # Q injection

        # --------- conductance G_k -----------------
        data.append(V2[k])    # ∂P/∂G_k = |V|²
        rows.append(rP)
        cols.append(col_G)
        # Q‑row derivative is zero → skip

        # --------- susceptance B_k -----------------
        data.append(-V2[k])   # ∂Q/∂B_k = −|V|²
        rows.append(rQ)
        cols.append(col_B)
        # P‑row derivative is zero → skip

    return csr_matrix((data, (rows, cols)), shape=(m_tot, n_par))
