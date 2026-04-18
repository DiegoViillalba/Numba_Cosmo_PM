"""
diagnostics.py
==============
Cálculo de energías y momento para verificar la correctitud física.

Equivalente de src/diagnostics.cpp en la versión C++.

Energía cinética:  Ek = Σ ½ m v²   (numpy: suma vectorizada)
Energía potencial: Ep = ½ Σ m φ(xi) (NGP interpolation)
Momento:           P  = Σ m v       (numpy: suma vectorizada)

En C++ las sumas sobre partículas requieren ``#pragma omp reduction``
para ser seguras en paralelo. En numpy las sumas son siempre seguras
porque numpy no usa múltiples hilos en np.sum() por defecto.
La equivalencia queda clara al comparar el código:

C++::

    double Ek = 0;
    #pragma omp parallel for reduction(+:Ek)
    for (size_t i = 0; i < N; i++)
        Ek += 0.5 * m[i] * (vx[i]**2 + vy[i]**2 + vz[i]**2);

Python::

    Ek = 0.5 * np.sum(m * (vel**2).sum(axis=1))
"""

from __future__ import annotations

import math
import numpy as np

from .types import SimState, Diagnostics
from . import config as cfg


def compute_diagnostics(state: SimState) -> Diagnostics:
    """
    Calcula energía cinética, potencial, momento y cociente virial.

    Returns
    -------
    Diagnostics namedtuple con campos:
        kinetic_energy, potential_energy, total_energy,
        momentum (array 3-D), virial_ratio
    """
    Ng    = cfg.NG
    p     = state.particles
    mass  = p["mass"]       # (N,)
    vel   = p["vel"]        # (N, 3)
    pos   = p["pos"]        # (N, 3)

    # ── Energía cinética: Ek = Σ ½ m v² ──────────────────────────
    v2 = np.sum(vel**2, axis=1)           # (N,) — suma sobre x,y,z
    Ek = 0.5 * np.dot(mass, v2)           # escalar

    # ── Momento total: P = Σ m·v ──────────────────────────────────
    momentum = mass[:, None] * vel        # (N, 3)
    P = momentum.sum(axis=0)              # (3,)

    # ── Energía potencial: Ep = ½ Σ m φ(xi) — interpolación NGP ──
    # Nearest Grid Point: celda más cercana
    ix = np.round(pos[:, 0]).astype(int) % Ng
    iy = np.round(pos[:, 1]).astype(int) % Ng
    iz = np.round(pos[:, 2]).astype(int) % Ng
    phi_at_part = state.potential[ix, iy, iz]    # (N,)
    Ep = 0.5 * np.dot(mass, phi_at_part)

    Etot = Ek + Ep
    virial = abs(Ep) / (2.0 * Ek) if Ek > 1e-30 else 0.0

    return Diagnostics(
        kinetic_energy   = float(Ek),
        potential_energy = float(Ep),
        total_energy     = float(Etot),
        momentum         = P,
        virial_ratio     = float(virial),
    )


# ──────────────────────────────────────────────────────────────────
# Impresión formateada — equivalente de print_diagnostics() en C++
# ──────────────────────────────────────────────────────────────────
_header_printed = False


def print_diagnostics(step: int, time: float, d: Diagnostics) -> None:
    """Imprime una línea de diagnóstico con formato tabular."""
    global _header_printed
    if not _header_printed:
        print()
        print(f"{'paso':>6}  {'tiempo':>8}  {'Ek':>14}  "
              f"{'Ep':>14}  {'Etot':>14}  {'|P|':>10}  {'virial':>8}")
        print("-" * 80)
        _header_printed = True

    P_mag = float(np.linalg.norm(d.momentum))
    print(
        f"{step:6d}  {time:8.4f}  "
        f"{d.kinetic_energy:14.6e}  {d.potential_energy:14.6e}  "
        f"{d.total_energy:14.6e}  {P_mag:10.3e}  {d.virial_ratio:8.3f}"
    )
