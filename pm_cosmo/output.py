"""
output.py
=========
Escritura de snapshots de partículas y archivo de diagnósticos CSV.

Equivalente de src/output.cpp en la versión C++.

Formatos
--------
ASCII  : primera línea = N_PART, luego ``x y z vx vy vz`` por partícula.
         Compatible con read_ic() de ambas versiones (Python y C++).
Binario: cabecera uint64 + volcado de Particle structs (7 doubles).
         Compatible byte a byte con write_snapshot_binary de C++.
CSV    : step, time, Ek, Ep, Etot, dE_frac — para graficar conservación.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from .types import SimState
from . import config as cfg


def write_snapshot_ascii(state: SimState, filename: str) -> None:
    """
    Escribe posiciones y velocidades en formato ASCII.

    Formato::

        N_PART
        x0 y0 z0 vx0 vy0 vz0
        ...
    """
    p    = state.particles
    data = np.column_stack([p["pos"], p["vel"]])   # (N, 6)

    with open(filename, "w") as f:
        f.write(f"{len(p)}\n")
        np.savetxt(f, data, fmt="%.8f")

    print(f"[output] Snapshot ASCII → {filename}  ({len(p)} partículas)")


def write_snapshot_binary(state: SimState, filename: str) -> None:
    """
    Escribe snapshot en formato binario compatible con la versión C++.

    El layout es: uint64 N_PART + N_PART × 7 doubles (x,y,z,vx,vy,vz,m).

    Para leer en Python::

        import numpy as np
        with open("snap_0010.bin", "rb") as f:
            n = np.frombuffer(f.read(8), dtype=np.uint64)[0]
            data = np.frombuffer(f.read(), dtype=np.float64).reshape(n, 7)
        pos = data[:, :3]
        vel = data[:, 3:6]
    """
    p    = state.particles
    n    = len(p)
    data = np.column_stack([p["pos"], p["vel"], p["mass"]])  # (N, 7)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", n))      # uint64 little-endian
        f.write(data.astype(np.float64).tobytes())

    size_mb = (8 + n * 7 * 8) / (1024 * 1024)
    print(f"[output] Snapshot binario → {filename}  ({n} partículas, {size_mb:.1f} MB)")


def append_diagnostics_csv(
    diag_file: str,
    step:      int,
    time:      float,
    Ek:        float,
    Ep:        float,
    Etot:      float,
) -> None:
    """
    Agrega una fila al CSV de evolución de energías.

    Crea el archivo con cabecera si no existe todavía.
    """
    needs_header = not Path(diag_file).exists()

    with open(diag_file, "a") as f:
        if needs_header:
            f.write("step,time,Ek,Ep,Etot,dE_frac\n")
        dE = (Etot - Ek - Ep) / abs(Etot) if abs(Etot) > 1e-30 else 0.0
        f.write(f"{step},{time:.6f},{Ek:.8e},{Ep:.8e},{Etot:.8e},{dE:.8e}\n")
