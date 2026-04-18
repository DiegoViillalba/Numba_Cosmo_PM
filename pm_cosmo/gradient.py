"""
gradient.py
===========
Calcula el campo gravitacional g = -∇φ por diferencias finitas centradas.

Equivalente de src/gradient.cpp en la versión C++.

El esquema de segundo orden es:
    g_x(i,j,k) = -(φ(i+1,j,k) - φ(i-1,j,k)) / (2·Δx)

Las condiciones de frontera periódicas se manejan con np.roll()
(versión numpy) o con índices módulo (versión numba).

Paralelismo
-----------
La versión numba usa prange con collapse implícito sobre los tres ejes,
lo que da un speedup casi lineal: cada celda es completamente independiente.
La versión numpy usa np.roll() que es implícitamente vectorizada.
"""

from __future__ import annotations

import numpy as np

from .types import SimState
from . import config as cfg

try:
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*a, **k):
        def d(fn): return fn
        return d
    prange = range


# ──────────────────────────────────────────────────────────────────
# Kernel numba — loop triple paralelo equivalente a collapse(3) en C++
# ──────────────────────────────────────────────────────────────────
@njit(parallel=True, cache=True)
def _gradient_kernel(
    potential: np.ndarray,  # (Ng, Ng, Ng)
    force_x:   np.ndarray,  # (Ng, Ng, Ng) — salida
    force_y:   np.ndarray,
    force_z:   np.ndarray,
    Ng:        int,
    inv2dx:    float,       # 1 / (2·Δx)
) -> None:
    """
    Kernel de diferencias finitas centradas paralelizado con numba.

    Cada iteración de prange sobre ix es independiente, y los loops
    internos sobre iy, iz tampoco tienen dependencias → no hay
    condición de carrera.
    """
    for ix in prange(Ng):     # paralelizado
        for iy in range(Ng):
            for iz in range(Ng):
                ixm = (ix - 1) % Ng;  ixp = (ix + 1) % Ng
                iym = (iy - 1) % Ng;  iyp = (iy + 1) % Ng
                izm = (iz - 1) % Ng;  izp = (iz + 1) % Ng

                force_x[ix, iy, iz] = -(potential[ixp, iy, iz]
                                        - potential[ixm, iy, iz]) * inv2dx
                force_y[ix, iy, iz] = -(potential[ix, iyp, iz]
                                        - potential[ix, iym, iz]) * inv2dx
                force_z[ix, iy, iz] = -(potential[ix, iy, izp]
                                        - potential[ix, iy, izm]) * inv2dx


def _gradient_numpy(
    potential: np.ndarray,
    force_x:   np.ndarray,
    force_y:   np.ndarray,
    force_z:   np.ndarray,
    inv2dx:    float,
) -> None:
    """
    Versión numpy del gradiente con np.roll() para condiciones periódicas.

    np.roll(a, -1, axis=0) desplaza el array un paso hacia adelante en el
    eje 0, lo que equivale a acceder al vecino i+1 con wraparound.
    """
    force_x[:] = -(np.roll(potential, -1, axis=0)
                   - np.roll(potential,  1, axis=0)) * inv2dx
    force_y[:] = -(np.roll(potential, -1, axis=1)
                   - np.roll(potential,  1, axis=1)) * inv2dx
    force_z[:] = -(np.roll(potential, -1, axis=2)
                   - np.roll(potential,  1, axis=2)) * inv2dx


# ──────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────
def compute_gradient(state: SimState) -> None:
    """
    Calcula g = -∇φ y almacena en state.force_x/y/z.

    Lee  state.potential  (Ng, Ng, Ng).
    Escribe state.force_x, state.force_y, state.force_z.
    """
    inv2dx = 1.0 / (2.0 * cfg.CELL_SIZE)

    if _NUMBA:
        _gradient_kernel(
            state.potential,
            state.force_x, state.force_y, state.force_z,
            cfg.NG, inv2dx,
        )
    else:
        _gradient_numpy(
            state.potential,
            state.force_x, state.force_y, state.force_z,
            inv2dx,
        )
