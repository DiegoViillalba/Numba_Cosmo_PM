"""
ic_reader.py
============
Lectura y generación de condiciones iniciales.

Equivalente de src/ic_reader.cpp en la versión C++.

Formatos soportados
-------------------
- ASCII: primera línea = N_PART, luego una línea por partícula con
  ``x y z vx vy vz``
- Binario: cabecera uint64 con N_PART, luego volcado de struct Particle
  (compatible byte a byte con la versión C++)

La función generate_uniform_ic() crea una grilla regular perturbada
sin requerir archivo externo, útil para pruebas rápidas.
"""

from __future__ import annotations

import struct
import numpy as np

from .types import SimState, PARTICLE_DTYPE, make_particles
from . import config as cfg


def read_ic(state: SimState, filename: str) -> None:
    """
    Lee condiciones iniciales desde un archivo ASCII.

    Formato::

        N_PART
        x0 y0 z0 vx0 vy0 vz0
        x1 y1 z1 vx1 vy1 vz1
        ...

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si el número de partículas no coincide con cfg.N_PARTICLES.
    """
    with open(filename, "r") as f:
        n_part = int(f.readline().strip())

    if n_part != cfg.N_PARTICLES:
        raise ValueError(
            f"ic_reader: el archivo tiene {n_part} partículas, "
            f"pero la simulación espera {cfg.N_PARTICLES} (Ng={cfg.NG}).\n"
            f"  Solución: regenera el IC con Ng={cfg.NG}."
        )

    # Leer todos los datos en una pasada (mucho más rápido que línea a línea)
    raw = np.loadtxt(filename, skiprows=1, dtype=np.float64)  # shape (N, 6)

    state.particles = make_particles(n_part)
    state.particles["pos"] = raw[:, :3]
    state.particles["vel"] = raw[:, 3:6]
    state.particles["mass"] = 1.0 / n_part

    print(f"[ic_reader] Leídas {n_part} partículas desde '{filename}'")


def read_ic_binary(state: SimState, filename: str) -> None:
    """
    Lee condiciones iniciales en formato binario compatible con C++.

    El archivo empieza con un uint64 que indica N_PART, seguido del
    volcado de N_PART structs Particle (7 doubles cada uno).
    """
    with open(filename, "rb") as f:
        n_part = struct.unpack("<Q", f.read(8))[0]

        if n_part != cfg.N_PARTICLES:
            raise ValueError(
                f"ic_reader: el archivo tiene {n_part} partículas, "
                f"esperadas {cfg.N_PARTICLES}."
            )

        raw = np.frombuffer(f.read(), dtype=np.float64).reshape(n_part, 7)

    state.particles = make_particles(n_part)
    state.particles["pos"]  = raw[:, :3]
    state.particles["vel"]  = raw[:, 3:6]
    state.particles["mass"] = raw[:, 6]

    print(f"[ic_reader] Leídas {n_part} partículas (binario) desde '{filename}'")


def generate_uniform_ic(
    state:     SimState,
    sigma_pos: float = 0.5,
    sigma_vel: float = 0.1,
    seed:      int   = 42,
) -> None:
    """
    Genera condiciones iniciales como grilla regular perturbada.

    Coloca Np^3 partículas en centros de celda con desplazamientos y
    velocidades gaussianos pequeños. Equivalente de generate_uniform_ic()
    en ic_reader.cpp.

    Parameters
    ----------
    state     : SimState a inicializar
    sigma_pos : desviación estándar del desplazamiento (en unidades de celda)
    sigma_vel : desviación estándar de la velocidad inicial
    seed      : semilla del RNG (reproducibilidad)
    """
    rng = np.random.default_rng(seed)

    N   = cfg.NG
    Np  = cfg.NP
    spacing = N / Np

    # Grilla regular de centros de celda: (i + 0.5) * spacing
    ix, iy, iz = np.meshgrid(
        np.arange(Np, dtype=np.float64),
        np.arange(Np, dtype=np.float64),
        np.arange(Np, dtype=np.float64),
        indexing="ij",
    )
    # Aplanar en orden C (row-major), igual que los loops anidados del C++
    ix = ix.ravel()
    iy = iy.ravel()
    iz = iz.ravel()

    n_part = cfg.N_PARTICLES
    state.particles = make_particles(n_part)

    # Posiciones: centro de celda + perturbación gaussiana con módulo periódico
    pos = np.column_stack([
        (ix + 0.5) * spacing + rng.normal(0, sigma_pos, n_part),
        (iy + 0.5) * spacing + rng.normal(0, sigma_pos, n_part),
        (iz + 0.5) * spacing + rng.normal(0, sigma_pos, n_part),
    ])
    # Condiciones periódicas
    state.particles["pos"] = pos % N

    # Velocidades gaussianas
    state.particles["vel"] = rng.normal(0, sigma_vel, (n_part, 3))

    # Masa uniforme
    state.particles["mass"] = 1.0 / n_part

    print(
        f"[ic_reader] Generadas {n_part} partículas en grilla uniforme perturbada"
        f" (Ng={N}, seed={seed})"
    )
