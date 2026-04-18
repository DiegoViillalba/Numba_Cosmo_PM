"""
force_interp.py
===============
Interpolación trilineal (CIC inverso) de la fuerza a cada partícula.

Equivalente de src/force_interp.cpp en la versión C++.

Esta operación es el adjunto exacto del depósito CIC: en lugar de
distribuir masa de partícula → malla, distribuye fuerza de malla →
partícula usando los mismos pesos trilineales.

Paralelismo
-----------
Solo hay lectura en las mallas de fuerza y escritura en accel[p] que
es privado de cada iteración → sin condición de carrera.
prange sobre el loop de partículas da speedup casi lineal.
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
# Kernel numba — loop paralelo sobre partículas
# ──────────────────────────────────────────────────────────────────
@njit(parallel=True, cache=True)
def _interp_kernel(
    pos:     np.ndarray,    # (N, 3)
    fx:      np.ndarray,    # (Ng, Ng, Ng)
    fy:      np.ndarray,
    fz:      np.ndarray,
    accel:   np.ndarray,    # (N, 3) — salida
    Ng:      int,
) -> None:
    """
    Kernel de interpolación CIC paralelizado con numba.prange.

    Cada partícula p lee de fx/fy/fz (solo lectura) y escribe en
    accel[p] (posición única) → sin condición de carrera.
    """
    N = pos.shape[0]

    for p in prange(N):
        # Misma convención de desplazamiento que en el depósito
        px = pos[p, 0] - 0.5
        py = pos[p, 1] - 0.5
        pz = pos[p, 2] - 0.5

        i0 = int(np.floor(px))
        j0 = int(np.floor(py))
        k0 = int(np.floor(pz))

        dx = px - i0;  tx = 1.0 - dx
        dy = py - j0;  ty = 1.0 - dy
        dz = pz - k0;  tz = 1.0 - dz

        i0w = i0 % Ng;  i1 = (i0 + 1) % Ng
        j0w = j0 % Ng;  j1 = (j0 + 1) % Ng
        k0w = k0 % Ng;  k1 = (k0 + 1) % Ng

        # Pesos de las 8 celdas
        w000 = tx * ty * tz;  w100 = dx * ty * tz
        w010 = tx * dy * tz;  w110 = dx * dy * tz
        w001 = tx * ty * dz;  w101 = dx * ty * dz
        w011 = tx * dy * dz;  w111 = dx * dy * dz

        accel[p, 0] = (fx[i0w,j0w,k0w]*w000 + fx[i1,j0w,k0w]*w100 +
                       fx[i0w,j1, k0w]*w010 + fx[i1,j1, k0w]*w110 +
                       fx[i0w,j0w,k1 ]*w001 + fx[i1,j0w,k1 ]*w101 +
                       fx[i0w,j1, k1 ]*w011 + fx[i1,j1, k1 ]*w111)

        accel[p, 1] = (fy[i0w,j0w,k0w]*w000 + fy[i1,j0w,k0w]*w100 +
                       fy[i0w,j1, k0w]*w010 + fy[i1,j1, k0w]*w110 +
                       fy[i0w,j0w,k1 ]*w001 + fy[i1,j0w,k1 ]*w101 +
                       fy[i0w,j1, k1 ]*w011 + fy[i1,j1, k1 ]*w111)

        accel[p, 2] = (fz[i0w,j0w,k0w]*w000 + fz[i1,j0w,k0w]*w100 +
                       fz[i0w,j1, k0w]*w010 + fz[i1,j1, k0w]*w110 +
                       fz[i0w,j0w,k1 ]*w001 + fz[i1,j0w,k1 ]*w101 +
                       fz[i0w,j1, k1 ]*w011 + fz[i1,j1, k1 ]*w111)


def _interp_numpy(
    pos:   np.ndarray,
    fx:    np.ndarray,
    fy:    np.ndarray,
    fz:    np.ndarray,
    accel: np.ndarray,
    Ng:    int,
) -> None:
    """Versión numpy vectorizada del CIC inverso (fallback sin numba)."""
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

    for comp, f in enumerate([fx, fy, fz]):
        accel[:, comp] = (
            f[i0w,j0w,k0w] * tx*ty*tz + f[i1,j0w,k0w] * dx*ty*tz +
            f[i0w,j1, k0w] * tx*dy*tz + f[i1,j1, k0w] * dx*dy*tz +
            f[i0w,j0w,k1 ] * tx*ty*dz + f[i1,j0w,k1 ] * dx*ty*dz +
            f[i0w,j1, k1 ] * tx*dy*dz + f[i1,j1, k1 ] * dx*dy*dz
        )


# ──────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────
def interpolate_force(state: SimState) -> np.ndarray:
    """
    Interpola la fuerza de la malla a cada partícula (CIC inverso).

    Lee  state.force_x/y/z  y  state.particles["pos"].
    Devuelve accel: ndarray shape (N_PART, 3).
    """
    Ng    = cfg.NG
    N     = len(state.particles)
    accel = np.zeros((N, 3), dtype=np.float64)
    pos   = state.particles["pos"]   # view

    if _NUMBA:
        _interp_kernel(
            pos,
            state.force_x, state.force_y, state.force_z,
            accel, Ng,
        )
    else:
        _interp_numpy(
            pos,
            state.force_x, state.force_y, state.force_z,
            accel, Ng,
        )

    return accel
