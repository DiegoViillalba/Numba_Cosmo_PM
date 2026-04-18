"""
config.py
=========
Parámetros globales de la simulación Particle-Mesh.

Equivalente directo de include/config.hpp en la versión C++.

Convención de unidades
----------------------
- Longitud : unidades de celda, posiciones en [0, Ng)
- Masa     : fracción de la masa total (suma de todas las partículas = 1)
- G        : 1.0 (unidades internas)

Para cambiar entre modo prueba (128) y producción (256) basta con
modificar NG en este archivo o sobreescribirlo desde la línea de comandos:

    python main.py --ng 128 --nsteps 20
"""

import os

# ──────────────────────────────────────────────────────────────────
# Tamaño de malla y partículas
# ──────────────────────────────────────────────────────────────────
NG: int = int(os.environ.get("PM_GRID_SIZE", 128))  # puntos de malla por eje
NP: int = NG                                          # partículas por eje
N_PARTICLES: int = NP ** 3                            # total de partículas
N_CELLS: int = NG ** 3                                # total de celdas

# ──────────────────────────────────────────────────────────────────
# Geometría de la caja
# ──────────────────────────────────────────────────────────────────
L_BOX: float = 100.0                # tamaño físico en Mpc/h
CELL_SIZE: float = L_BOX / NG       # tamaño de celda en Mpc/h

# ──────────────────────────────────────────────────────────────────
# Integración temporal
# ──────────────────────────────────────────────────────────────────
DT: float = 0.01        # paso de tiempo en unidades internas
N_STEPS: int = 100      # número de pasos por defecto

# ──────────────────────────────────────────────────────────────────
# Constante gravitacional (unidades internas G = 1)
# ──────────────────────────────────────────────────────────────────
G_GRAV: float = 1.0

# ──────────────────────────────────────────────────────────────────
# Salida
# ──────────────────────────────────────────────────────────────────
SNAP_INTERVAL: int = 10   # cada cuántos pasos se escribe un snapshot

# ──────────────────────────────────────────────────────────────────
# Softening gravitacional
# ──────────────────────────────────────────────────────────────────
SOFTENING: float = CELL_SIZE / 50.0
