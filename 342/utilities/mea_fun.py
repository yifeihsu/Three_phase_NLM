import numpy as np

def measurement_function(
    x: np.ndarray,
    Ybus: np.ndarray,
    mpc: dict,
    busphase_map: dict[int, int],
) -> np.ndarray:
    nnode = len(busphase_map)
    Vm, Va = x[:nnode], x[nnode:]

    V = Vm * np.exp(1j * Va)

    I = Ybus @ V

    S = V * np.conj(I)
    P = S.real
    Q = S.imag

    h = np.concatenate([P, Q, Vm])

    return h