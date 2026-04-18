"""
types.py
========
Estructuras de datos centrales del simulador PM.

Equivalente de include/types.hpp en la versión C++.

En Python usamos numpy estructurado (structured array) para las partículas
y arrays contiguos de float64 para las mallas, lo que garantiza el mismo
layout de memoria que el código C++ y permite leer archivos binarios
generados por la versión C++.

Notas de diseño
---------------
- SimState agrupa todo el estado mutable de la simulación.
- Las mallas son arrays 3-D de shape (Ng, Ng, Ng) en row-major (C order).
  El índice plano de (ix, iy, iz) es ix*Ng*Ng + iy*Ng + iz, idéntico a C++.
- particle_dtype define el mismo layout que struct Particle en types.hpp,
  lo que permite intercambio binario directo entre versiones.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from . import config as cfg

# ──────────────────────────────────────────────────────────────────
# Dtype de partícula — mismo layout que struct Particle en C++
# (pos[3], vel[3], mass) → 7 doubles = 56 bytes
# ──────────────────────────────────────────────────────────────────
PARTICLE_DTYPE = np.dtype([
    ("pos",  np.float64, (3,)),   # posición en unidades de celda [0, Ng)
    ("vel",  np.float64, (3,)),   # velocidad en unidades internas
    ("mass", np.float64),         # masa (suma total = 1)
])


def make_particles(n: int | None = None) -> np.ndarray:
    """Crea un array de partículas inicializado a cero."""
    n = n or cfg.N_PARTICLES
    arr = np.zeros(n, dtype=PARTICLE_DTYPE)
    arr["mass"] = 1.0 / n
    return arr


def make_grid() -> np.ndarray:
    """Crea una malla 3-D (Ng, Ng, Ng) de float64 inicializada a cero."""
    return np.zeros((cfg.NG, cfg.NG, cfg.NG), dtype=np.float64)


# ──────────────────────────────────────────────────────────────────
# SimState — equivalente de struct SimState en C++
# ──────────────────────────────────────────────────────────────────
@dataclass
class SimState:
    """
    Agrupa todo el estado mutable de la simulación.

    Attributes
    ----------
    particles : ndarray, shape (N_PART,), dtype PARTICLE_DTYPE
    density   : ndarray, shape (Ng, Ng, Ng)  — campo ρ(x)
    potential : ndarray, shape (Ng, Ng, Ng)  — potencial φ(x)
    force_x   : ndarray, shape (Ng, Ng, Ng)  — componente x de g = -∇φ
    force_y   : ndarray, shape (Ng, Ng, Ng)
    force_z   : ndarray, shape (Ng, Ng, Ng)
    time      : float    — tiempo físico actual
    step      : int      — número de paso actual
    """
    particles: np.ndarray = field(default_factory=make_particles)
    density:   np.ndarray = field(default_factory=make_grid)
    potential: np.ndarray = field(default_factory=make_grid)
    force_x:   np.ndarray = field(default_factory=make_grid)
    force_y:   np.ndarray = field(default_factory=make_grid)
    force_z:   np.ndarray = field(default_factory=make_grid)
    time:  float = 0.0
    step:  int   = 0

    def clear_grids(self) -> None:
        """Pone a cero las mallas de densidad, potencial y fuerza."""
        self.density[:]   = 0.0
        self.potential[:] = 0.0
        self.force_x[:]   = 0.0
        self.force_y[:]   = 0.0
        self.force_z[:]   = 0.0


# ──────────────────────────────────────────────────────────────────
# Diagnostics — equivalente de struct Diagnostics en C++
# ──────────────────────────────────────────────────────────────────
class Diagnostics(NamedTuple):
    """Resultados de un cálculo de diagnósticos en un paso dado."""
    kinetic_energy:   float
    potential_energy: float
    total_energy:     float
    momentum:         np.ndarray   # shape (3,)
    virial_ratio:     float
