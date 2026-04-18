"""
test_ic_reader.py
=================
Tests para el módulo ic_reader.py.

Propiedades verificadas:
    1. generate_uniform_ic: N_PARTICLES partículas en [0, Ng).
    2. Masa total = 1 después de generar.
    3. Escritura + lectura ASCII produce el mismo estado.
    4. Error correcto cuando el archivo no existe.
    5. Sin sigma: posiciones exactamente en centros de celda.
"""

import os
import tempfile

import numpy as np
import pytest

from pm_cosmo.types import SimState
from pm_cosmo.ic_reader import generate_uniform_ic, read_ic
from pm_cosmo.output import write_snapshot_ascii
from pm_cosmo import config as cfg


# ─────────────────────────────────────────────────────────────────
# Test 1: número de partículas y dominio
# ─────────────────────────────────────────────────────────────────
def test_generate_n_particles():
    """generate_uniform_ic produce exactamente N_PARTICLES partículas."""
    state = SimState()
    generate_uniform_ic(state)

    assert len(state.particles) == cfg.N_PARTICLES


def test_generate_posiciones_en_dominio():
    """Todas las posiciones caen en [0, Ng)."""
    state = SimState()
    generate_uniform_ic(state, sigma_pos=2.0)  # perturbación grande

    pos = state.particles["pos"]
    assert np.all(pos >= 0.0), f"Posición mínima: {pos.min():.4f}"
    assert np.all(pos <  cfg.NG), f"Posición máxima: {pos.max():.4f}"


# ─────────────────────────────────────────────────────────────────
# Test 2: conservación de masa
# ─────────────────────────────────────────────────────────────────
def test_generate_masa_total():
    """Suma de masas == 1 (en unidades internas)."""
    state = SimState()
    generate_uniform_ic(state)

    total_mass = state.particles["mass"].sum()
    assert total_mass == pytest.approx(1.0, rel=1e-10)


# ─────────────────────────────────────────────────────────────────
# Test 3: round-trip ASCII — escribir y leer devuelve el mismo estado
# ─────────────────────────────────────────────────────────────────
def test_roundtrip_ascii(tmp_path):
    """write_snapshot_ascii + read_ic reproduce posiciones y velocidades."""
    state = SimState()
    generate_uniform_ic(state, sigma_pos=0.3, sigma_vel=0.05, seed=1)

    snap_file = str(tmp_path / "snap.dat")
    write_snapshot_ascii(state, snap_file)

    state2 = SimState()
    read_ic(state2, snap_file)

    # El formato ASCII usa %.8f → error máximo ~5e-9
    np.testing.assert_allclose(
        state2.particles["pos"], state.particles["pos"],
        atol=1e-7, rtol=0,
    )
    np.testing.assert_allclose(
        state2.particles["vel"], state.particles["vel"],
        atol=1e-7, rtol=0,
    )


# ─────────────────────────────────────────────────────────────────
# Test 4: FileNotFoundError para archivo inexistente
# ─────────────────────────────────────────────────────────────────
def test_read_ic_archivo_inexistente():
    """read_ic lanza FileNotFoundError si el archivo no existe."""
    state = SimState()
    with pytest.raises(FileNotFoundError):
        read_ic(state, "/tmp/archivo_que_no_existe_pm_cosmo.dat")


# ─────────────────────────────────────────────────────────────────
# Test 5: sin perturbación → posiciones en centros de celda exactos
# ─────────────────────────────────────────────────────────────────
def test_generate_sin_sigma_posiciones_exactas():
    """Con sigma_pos=0, las posiciones están en centros de celda exactos."""
    state = SimState()
    generate_uniform_ic(state, sigma_pos=0.0, sigma_vel=0.0)

    pos = state.particles["pos"]
    spacing = float(cfg.NG) / cfg.NP

    # Los centros de celda son (i + 0.5) * spacing para i en [0, Np)
    # Por lo tanto pos % spacing debe estar cerca de 0.5 * spacing
    residual = pos % spacing
    np.testing.assert_allclose(
        residual, 0.5 * spacing, atol=1e-10,
        err_msg="Posiciones no están en centros de celda"
    )


# ─────────────────────────────────────────────────────────────────
# Test 6: reproducibilidad con misma semilla
# ─────────────────────────────────────────────────────────────────
def test_generate_reproducible():
    """Misma semilla → mismas posiciones y velocidades."""
    s1 = SimState()
    s2 = SimState()
    generate_uniform_ic(s1, seed=42)
    generate_uniform_ic(s2, seed=42)

    np.testing.assert_array_equal(s1.particles["pos"], s2.particles["pos"])
    np.testing.assert_array_equal(s1.particles["vel"], s2.particles["vel"])
