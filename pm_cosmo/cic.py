"""
cic.py
======
Cloud-in-Cell (CIC): deposita la masa de las partículas en la malla.

Paralelismo y correctitud
--------------------------
El loop CIC tiene una condición de carrera cuando se paraleliza:
múltiples partículas pueden escribir en las mismas celdas simultáneamente.

Solución: acumuladores locales por hilo usando ``get_thread_id()`` de numba.
Cada hilo escribe en ``local[thread_id, ...]``, sin contención.
La reducción final es secuencial (suma los acumuladores).

Alternativa numpy: np.add.at() — siempre correcto, más lento.
"""

from __future__ import annotations
import numpy as np
from .types import SimState
from . import config as cfg

try:
    from numba import njit, prange, get_num_threads, get_thread_id
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*a, **k):
        def d(fn): return fn
        return d
    prange = range


@njit(parallel=True, cache=True)
def _cic_kernel(
    pos:      np.ndarray,   # (N, 3)
    mass:     np.ndarray,   # (N,)
    density:  np.ndarray,   # (Ng, Ng, Ng) — salida
    Ng:       int,
    n_threads: int,
) -> None:
    """
    CIC paralelo con acumuladores locales por hilo.

    get_thread_id() devuelve el ID real del hilo OpenMP subyacente
    dentro de un bloque prange → sin race condition.
    """
    N = pos.shape[0]
    # Acumulador local: un arreglo Ng³ por hilo
    local = np.zeros((n_threads, Ng, Ng, Ng), dtype=np.float64)

    for p in prange(N):
        tid = get_thread_id()   # ID real del hilo actual (0..n_threads-1)

        px = pos[p, 0] - 0.5
        py = pos[p, 1] - 0.5
        pz = pos[p, 2] - 0.5
        m  = mass[p]

        i0 = int(np.floor(px))
        j0 = int(np.floor(py))
        k0 = int(np.floor(pz))

        dx = px - i0;  tx = 1.0 - dx
        dy = py - j0;  ty = 1.0 - dy
        dz = pz - k0;  tz = 1.0 - dz

        i0w = i0 % Ng;  i1 = (i0 + 1) % Ng
        j0w = j0 % Ng;  j1 = (j0 + 1) % Ng
        k0w = k0 % Ng;  k1 = (k0 + 1) % Ng

        local[tid, i0w, j0w, k0w] += m * tx * ty * tz
        local[tid, i1,  j0w, k0w] += m * dx * ty * tz
        local[tid, i0w, j1,  k0w] += m * tx * dy * tz
        local[tid, i1,  j1,  k0w] += m * dx * dy * tz
        local[tid, i0w, j0w, k1 ] += m * tx * ty * dz
        local[tid, i1,  j0w, k1 ] += m * dx * ty * dz
        local[tid, i0w, j1,  k1 ] += m * tx * dy * dz
        local[tid, i1,  j1,  k1 ] += m * dx * dy * dz

    # Reducción: sumar acumuladores locales → density
    for t in range(n_threads):
        density += local[t]


def _cic_numpy(
    pos:     np.ndarray,
    mass:    np.ndarray,
    density: np.ndarray,
    Ng:      int,
) -> None:
    """
    CIC numpy con np.add.at() — correcto para índices repetidos.
    Equivalente a #pragma omp atomic en C++.
    """
    px = pos[:, 0] - 0.5
    py = pos[:, 1] - 0.5
    pz = pos[:, 2] - 0.5

    i0 = np.floor(px).astype(int)
    j0 = np.floor(py).astype(int)
    k0 = np.floor(pz).astype(int)

    dx = px - i0;  tx = 1.0 - dx
    dy = py - j0;  ty = 1.0 - dy
    dz = pz - k0;  tz = 1.0 - dz

    i0w = i0 % Ng;  i1 = (i0 + 1) % Ng
    j0w = j0 % Ng;  j1 = (j0 + 1) % Ng
    k0w = k0 % Ng;  k1 = (k0 + 1) % Ng

    for ii, jj, kk, w in [
        (i0w, j0w, k0w, tx * ty * tz),
        (i1,  j0w, k0w, dx * ty * tz),
        (i0w, j1,  k0w, tx * dy * tz),
        (i1,  j1,  k0w, dx * dy * tz),
        (i0w, j0w, k1,  tx * ty * dz),
        (i1,  j0w, k1,  dx * ty * dz),
        (i0w, j1,  k1,  tx * dy * dz),
        (i1,  j1,  k1,  dx * dy * dz),
    ]:
        np.add.at(density, (ii, jj, kk), mass * w)


def cic_deposit(state: SimState) -> None:
    """
    Deposita la masa de las partículas en state.density usando CIC.

    Normaliza al final: suma total = N_CELLS (densidad media = 1).
    """
    Ng   = cfg.NG
    pos  = state.particles["pos"]
    mass = state.particles["mass"]

    state.density[:] = 0.0

    if _NUMBA:
        n_threads = get_num_threads()
        _cic_kernel(pos, mass, state.density, Ng, n_threads)
    else:
        _cic_numpy(pos, mass, state.density, Ng)

    state.density *= cfg.N_CELLS
