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
import time
from pathlib import Path

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from .types import SimState
from . import config as cfg


def _open_h5_for_write(filename: str, retries: int = 3):
    """
    Abre un archivo HDF5 para escritura con tolerancia a bloqueos.

    En algunos sistemas (incluyendo macOS) puede aparecer errno 35
    ("Resource temporarily unavailable") si otro proceso tiene un lock.
    """
    last_err: OSError | None = None
    for attempt in range(retries):
        try:
            # h5py >= 3.8 soporta controlar locking explícitamente.
            return h5py.File(filename, "w", locking=False)
        except TypeError:
            # h5py antiguo: fallback sin parámetro locking.
            pass
        except OSError as exc:
            if getattr(exc, "errno", None) == 35:
                last_err = exc
                # Fallback para builds HDF5 que respetan la variable de entorno.
                os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
                time.sleep(0.05 * (attempt + 1))
                continue
            raise

        try:
            return h5py.File(filename, "w")
        except OSError as exc:
            if getattr(exc, "errno", None) == 35:
                last_err = exc
                os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
                time.sleep(0.05 * (attempt + 1))
                continue
            raise

    raise OSError(
        35,
        (
            f"No se pudo crear {filename} por bloqueo de archivo HDF5. "
            "Cierra GadgetViewer/u otros lectores que tengan abierto el snapshot "
            "o cambia el nombre de salida."
        ),
    ) from last_err


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


def write_snapshot_hdf5_gadget(
    state:    SimState,
    filename: str,
    step:     int | None = None,
) -> None:
    """
        Escribe snapshot en formato HDF5 compatible con Gadget/GadgetViewer.

        Notas de compatibilidad:
        - Gadget espera un grupo ``Header`` (no atributos en la raíz).
        - Materia oscura colisionalmente fría se guarda en ``PartType1``.
        - Se incluyen los atributos requeridos del header (NumFilesPerSnapshot,
            NumPart_ThisFile, NumPart_Total, etc.).
    
    Args:
        state: SimState con partículas
        filename: ruta del archivo a escribir
        step: número de paso (opcional, incluido en metadatos)
    
    Raises:
        ImportError: si h5py no está disponible
    """
    if not HAS_H5PY:
        raise ImportError(
            "h5py es requerido para escribir snapshots HDF5. "
            "Instala con: pip install h5py"
        )
    
    p = state.particles
    n = len(p)
    
    with _open_h5_for_write(filename) as f:
        # Header en grupo dedicado (formato Gadget HDF5)
        h = f.create_group("Header")
        header = h.attrs

        # En Gadget: type 1 = dark matter
        numpart = np.array([0, n, 0, 0, 0, 0], dtype=np.uint32)
        header["NumPart_ThisFile"] = numpart
        header["NumPart_Total"] = numpart
        header["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        header["MassTable"] = np.zeros(6, dtype=np.float64)  # masas individuales

        header["Time"] = np.float64(state.time)
        header["Redshift"] = np.float64(0.0)
        header["BoxSize"] = np.float64(cfg.GRID_SIZE)
        header["NumFilesPerSnapshot"] = np.int32(1)

        # Flags estándar que varios lectores esperan
        header["Flag_Sfr"] = np.int32(0)
        header["Flag_Cooling"] = np.int32(0)
        header["Flag_StellarAge"] = np.int32(0)
        header["Flag_Metals"] = np.int32(0)
        header["Flag_Feedback"] = np.int32(0)
        header["Flag_Entropy_ICs"] = np.int32(0)
        header["Flag_DoublePrecision"] = np.int32(1)

        # Cosmología nominal para visualización (ajustable)
        header["Omega0"] = np.float64(0.3)
        header["OmegaLambda"] = np.float64(0.7)
        header["HubbleParam"] = np.float64(0.7)

        if step is not None:
            header["SnapshotNumber"] = np.int32(step)

        # Grupo de partículas: PartType1 para DM
        dm = f.create_group("PartType1")
        
        # Posiciones [N, 3] en unidades de código
        coords = dm.create_dataset(
            "Coordinates",
            data=p["pos"].astype(np.float64),
            compression="gzip",
            compression_opts=4
        )
        coords.attrs["a"] = np.float32(1.0)      # factor de escala
        coords.attrs["h"] = np.float32(0.7)     # parámetro Hubble
        
        # Velocidades [N, 3] en unidades de código
        vels = dm.create_dataset(
            "Velocities",
            data=p["vel"].astype(np.float64),
            compression="gzip",
            compression_opts=4
        )
        vels.attrs["a"] = np.float32(1.0)
        vels.attrs["h"] = np.float32(0.7)
        
        # Masas individuales [N] en unidades de código
        masses = dm.create_dataset(
            "Masses",
            data=p["mass"].astype(np.float64),
            compression="gzip",
            compression_opts=4
        )
        masses.attrs["a"] = np.float32(1.0)
        masses.attrs["h"] = np.float32(0.7)
        
        # ParticleIDs: identificadores únicos [N] (consecutivos)
        ids = dm.create_dataset(
            "ParticleIDs",
            data=np.arange(1, n + 1, dtype=np.uint64),
            compression="gzip",
            compression_opts=4
        )
        ids.attrs["a"] = np.float32(1.0)
        ids.attrs["h"] = np.float32(0.7)
    
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(
        f"[output] Snapshot HDF5 (Gadget) → {filename}  "
        f"({n} partículas, {size_mb:.2f} MB)"
    )
