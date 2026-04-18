"""
poisson_fft.py
==============
Resolución de ∇²φ = 4πG ρ en espacio de Fourier.

Equivalente de src/poisson_fft.cpp en la versión C++.

Algoritmo
---------
1. FFT directa:    ρ(x)  → ρ̂(k)            [numpy.fft.rfftn]
2. Multiplicar:    φ̂(k) = -4πG ρ̂(k) / k²   [operación vectorizada]
3. FFT inversa:    φ̂(k) → φ(x)             [numpy.fft.irfftn]

numpy.fft.rfftn aprovecha la simetría hermítica del campo real para
reducir el almacenamiento a la mitad, igual que FFTW3 con r2c/c2r en C++.

Paralelismo
-----------
numpy.fft.rfftn puede usar múltiples hilos vía pyfftw o las librerías
BLAS subyacentes (MKL/OpenBLAS). La multiplicación por el kernel es
una operación vectorizada numpy que numpy puede paralelizar internamente.
Para paralelismo explícito del kernel se usa numba si está disponible.
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


def _build_k2_grid(Ng: int, cell_size: float) -> np.ndarray:
    """
    Precalcula la malla k² en espacio de Fourier para la FFT real (rfftn).

    Para una malla de tamaño Ng con paso físico Δx = cell_size:
        kx = (2π/Ng) × (ix  si ix ≤ Ng/2, sino ix - Ng)
        k_phys = k_discrete / Δx

    La malla de salida tiene shape (Ng, Ng, Ng//2+1), el mismo shape
    que la salida de numpy.fft.rfftn sobre una malla (Ng, Ng, Ng).
    """
    # Frecuencias normalizadas (en unidades discretas)
    freq = np.fft.fftfreq(Ng, d=1.0)            # shape (Ng,)
    freq_r = np.fft.rfftfreq(Ng, d=1.0)         # shape (Ng//2+1,) — para la dim z

    # Convertir a k físico: k = 2π × freq / Δx
    two_pi_over_dx = 2.0 * np.pi / cell_size
    kx = freq   * two_pi_over_dx
    ky = freq   * two_pi_over_dx
    kz = freq_r * two_pi_over_dx

    # Crear malla 3-D con broadcasting
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2    # shape (Ng, Ng, Ng//2+1)

    return k2


# Precalcular k² una sola vez (constante para toda la simulación)
_K2: np.ndarray | None = None


def _get_k2() -> np.ndarray:
    global _K2
    if _K2 is None:
        _K2 = _build_k2_grid(cfg.NG, cfg.CELL_SIZE)
    return _K2


def solve_poisson(state: SimState) -> None:
    """
    Resuelve ∇²φ = 4πGρ en espacio de Fourier.

    Lee  state.density  (Ng, Ng, Ng).
    Escribe state.potential (Ng, Ng, Ng).

    El modo k=0 (densidad media) se pone a cero, equivalente a trabajar
    con el contraste de densidad δ = ρ/ρ̄ - 1.
    """
    Ng = cfg.NG
    G4pi = 4.0 * np.pi * cfg.G_GRAV
    k2   = _get_k2()   # shape (Ng, Ng, Ng//2+1)

    # ── 1. FFT directa: ρ(x) → ρ̂(k) ─────────────────────────────
    rho_k = np.fft.rfftn(state.density)   # shape (Ng, Ng, Ng//2+1), complex128

    # ── 2. Multiplicar por el kernel de Green: φ̂ = -4πG ρ̂ / k² ──
    # Evitar división por cero en k=0; ese modo se anula al final
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = np.where(k2 > 1e-30, -G4pi / k2, 0.0)

    phi_k = rho_k * kernel    # broadcasting automático

    # Modo k=0: poner explícitamente a cero (densidad de fondo)
    phi_k[0, 0, 0] = 0.0

    # ── 3. FFT inversa: φ̂(k) → φ(x) ──────────────────────────────
    # irfftn normaliza automáticamente (divide por N_CELLS), igual que
    # el paso de normalización manual en el código C++.
    state.potential = np.fft.irfftn(phi_k, s=(Ng, Ng, Ng), axes=(0, 1, 2))
