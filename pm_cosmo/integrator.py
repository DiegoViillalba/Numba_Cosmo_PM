"""
integrator.py
=============
Integrador Leap-Frog (Verlet de velocidades desfasadas).

Equivalente de src/integrator.cpp en la versión C++.

El Leap-Frog es un integrador simpléctico de 2.° orden:
    v_{n+1/2} = v_{n-1/2} + a_n · Δt      (kick)
    x_{n+1}   = x_n + v_{n+1/2} · Δt      (drift)

Para aceleración constante reproduce x = ½·a·t² exactamente,
lo que se verifica en los tests.

Paralelismo
-----------
Cada partícula es completamente independiente de las demás:
solo lee accel[i] y escribe en particles[i] → sin condición de carrera.
prange sobre el loop de partículas da speedup lineal.
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
def _leapfrog_kernel(
    pos:   np.ndarray,   # (N, 3) — modificado in-place
    vel:   np.ndarray,   # (N, 3) — modificado in-place
    accel: np.ndarray,   # (N, 3) — solo lectura
    dt:    float,
    Ng:    float,
) -> None:
    """
    Un paso Leap-Frog para todas las partículas (kernel numba).

    kick : vel += accel · dt
    drift: pos += vel   · dt  (con wraparound periódico)
    """
    N = pos.shape[0]
    for i in prange(N):   # prange → paralelizado
        # Kick
        vel[i, 0] += accel[i, 0] * dt
        vel[i, 1] += accel[i, 1] * dt
        vel[i, 2] += accel[i, 2] * dt

        # Drift
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        pos[i, 2] += vel[i, 2] * dt

        # Condiciones de frontera periódicas
        # El módulo de Python es siempre positivo para Ng > 0, así que
        # pos % Ng siempre da un valor en [0, Ng).
        pos[i, 0] %= Ng
        pos[i, 1] %= Ng
        pos[i, 2] %= Ng


@njit(parallel=True, cache=True)
def _half_kick_kernel(
    vel:   np.ndarray,
    accel: np.ndarray,
    half_dt: float,
) -> None:
    """Medio kick inicial para arrancar el Leap-Frog desde v(t=0)."""
    N = vel.shape[0]
    for i in prange(N):
        vel[i, 0] -= accel[i, 0] * half_dt
        vel[i, 1] -= accel[i, 1] * half_dt
        vel[i, 2] -= accel[i, 2] * half_dt


# ──────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────
def leapfrog_step(state: SimState, accel: np.ndarray) -> None:
    """
    Un paso completo Leap-Frog: kick de velocidad + drift de posición.

    Modifica in-place state.particles["pos"] y state.particles["vel"].
    Actualiza state.time y state.step.

    Parameters
    ----------
    state : SimState actual
    accel : ndarray shape (N_PART, 3) — aceleración calculada por interpolate_force
    """
    pos = state.particles["pos"]   # view → modificación in-place
    vel = state.particles["vel"]
    Ng  = float(cfg.NG)
    dt  = cfg.DT

    if _NUMBA:
        _leapfrog_kernel(pos, vel, accel, dt, Ng)
    else:
        # Versión numpy vectorizada (fallback)
        vel += accel * dt
        pos += vel   * dt
        pos %= Ng

    state.time += dt
    state.step += 1


def leapfrog_half_kick(state: SimState, accel: np.ndarray) -> None:
    """
    Medio kick inicial: v_{-1/2} = v_0 - a_0 · (Δt/2).

    Se llama UNA SOLA VEZ antes del primer paso completo para sincronizar
    las velocidades al tiempo t = -Δt/2, que es el offset requerido por
    el esquema Leap-Frog cuando las IC dan v en t=0.
    """
    vel     = state.particles["vel"]
    half_dt = cfg.DT * 0.5

    if _NUMBA:
        _half_kick_kernel(vel, accel, half_dt)
    else:
        vel -= accel * half_dt
