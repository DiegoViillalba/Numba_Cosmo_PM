"""
test_cic.py
===========
Tests para el módulo cic.py (Cloud-in-Cell deposit).

Propiedades verificadas — espejo exacto de test_cic.cpp:
    1. Conservación de masa: suma(density) == N_CELLS tras normalizar.
    2. Split 50/50 en el borde entre dos celdas.
    3. No negativos: density >= 0 en todas las celdas.
    4. Grilla regular → densidad uniforme ≈ 1.
"""

import numpy as np
import pytest

from pm_cosmo.types import SimState, make_particles
from pm_cosmo.cic import cic_deposit
from pm_cosmo import config as cfg


# ─────────────────────────────────────────────────────────────────
# Helpers locales
# ─────────────────────────────────────────────────────────────────

def _state_all_at(pos_xyz, n: int = None) -> SimState:
    """SimState con n partículas iguales todas en pos_xyz."""
    n = n or cfg.N_PARTICLES
    state = SimState()
    state.particles = make_particles(n)
    state.particles["pos"] = pos_xyz
    state.particles["vel"] = 0.0
    return state


def _single(x, y, z, mass=1.0) -> SimState:
    """SimState con una sola partícula en (x, y, z) de masa dada."""
    state = SimState()
    state.particles = make_particles(1)
    state.particles["pos"][0] = [x, y, z]
    state.particles["vel"][0] = [0.0, 0.0, 0.0]
    state.particles["mass"][0] = mass
    return state


# ─────────────────────────────────────────────────────────────────
# Test 1: conservación de masa
# suma(density) debe ser N_CELLS después de la normalización.
# ─────────────────────────────────────────────────────────────────
def test_cic_conservacion_masa():
    """Suma total de density == N_CELLS (conservación de masa)."""
    state = _state_all_at([1.0, 1.0, 1.0], n=cfg.N_PARTICLES)
    cic_deposit(state)

    total = state.density.sum()
    np.testing.assert_allclose(
        total, cfg.N_CELLS,
        rtol=1e-6,
        err_msg=f"Suma de density = {total}, esperado {cfg.N_CELLS}",
    )


# ─────────────────────────────────────────────────────────────────
# Test 2: split 50/50 en el borde entre celdas
# Partícula en x=5.0 (exactamente en la frontera celda 4/5) debe
# depositar pesos iguales en ambas celdas.
# ─────────────────────────────────────────────────────────────────
def test_cic_split_50_50():
    """Partícula en borde de celda → densidad igual en las dos celdas adyacentes."""
    state = _single(5.0, 1.5, 1.5)
    cic_deposit(state)

    # Celda base: floor(5.0 - 0.5) = 4
    d4 = state.density[4, 1, 1]
    d5 = state.density[5, 1, 1]

    assert d4 > 0, "Celda 4 no recibió masa"
    np.testing.assert_allclose(
        d4, d5, rtol=1e-10,
        err_msg=f"Split no es 50/50: d4={d4}, d5={d5}",
    )


# ─────────────────────────────────────────────────────────────────
# Test 3: no negativos
# CIC con pesos trilineales ∈ [0,1] garantiza density >= 0 siempre.
# ─────────────────────────────────────────────────────────────────
def test_cic_no_negativos():
    """Todas las celdas deben tener densidad >= 0."""
    state = SimState()
    state.particles = make_particles(200)
    # Posiciones pseudoaleatorias deterministas
    idx = np.arange(200, dtype=np.float64)
    state.particles["pos"][:, 0] = (idx * 7.31)  % cfg.NG
    state.particles["pos"][:, 1] = (idx * 3.71)  % cfg.NG
    state.particles["pos"][:, 2] = (idx * 11.13) % cfg.NG
    state.particles["vel"]       = 0.0

    cic_deposit(state)

    assert np.all(state.density >= -1e-12), (
        f"Densidad negativa mínima: {state.density.min():.3e}"
    )


# ─────────────────────────────────────────────────────────────────
# Test 4: grilla regular → densidad uniforme ≈ 1
# Partículas en centros exactos de celda deben dar density ≈ 1
# en todas partes.
# ─────────────────────────────────────────────────────────────────
def test_cic_grilla_uniforme():
    """Grilla regular perfecta → densidad uniforme ≈ 1 en todo el dominio."""
    from pm_cosmo.ic_reader import generate_uniform_ic
    state = SimState()
    generate_uniform_ic(state, sigma_pos=0.0, sigma_vel=0.0)

    cic_deposit(state)

    np.testing.assert_allclose(
        state.density,
        np.ones_like(state.density),
        atol=0.01,
        err_msg=(
            f"Densidad no uniforme: min={state.density.min():.4f}, "
            f"max={state.density.max():.4f}"
        ),
    )


# ─────────────────────────────────────────────────────────────────
# Test 5: idempotencia del zero — limpiar y re-depositar da el mismo resultado
# ─────────────────────────────────────────────────────────────────
def test_cic_idempotente():
    """Dos depósitos consecutivos con clear_grids dan resultados idénticos."""
    from pm_cosmo.ic_reader import generate_uniform_ic
    state = SimState()
    generate_uniform_ic(state, sigma_pos=0.3, sigma_vel=0.0, seed=7)

    cic_deposit(state)
    d1 = state.density.copy()

    state.clear_grids()
    cic_deposit(state)
    d2 = state.density.copy()

    np.testing.assert_array_equal(d1, d2,
        err_msg="Segundo depósito produce resultado diferente al primero")
