"""
test_diagnostics.py
===================
Tests para el módulo diagnostics.py.

Propiedades verificadas — espejo de test_diagnostics.cpp:
    1. Ek = ½·m·v² para partícula única.
    2. Momento cero con velocidades opuestas.
    3. Reposo → Ek = 0, |P| = 0.
    4. Etot == Ek + Ep.
    5. Ek escala linealmente con masa.
    6. (Extra) Cociente virial = |Ep| / (2·Ek).
"""

import numpy as np
import pytest

from pm_cosmo.types import SimState, make_particles
from pm_cosmo.diagnostics import compute_diagnostics
from pm_cosmo import config as cfg


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _state_with_particles(n: int, vels, pos=None) -> SimState:
    """Crea un SimState con n partículas y velocidades dadas."""
    state = SimState()
    state.particles = make_particles(n)
    state.particles["vel"][:] = vels
    if pos is not None:
        state.particles["pos"][:] = pos
    return state


# ─────────────────────────────────────────────────────────────────
# Test 1: Ek = ½·m·v² para partícula única
# ─────────────────────────────────────────────────────────────────
def test_diag_ek_particula_unica():
    """Ek = ½·m·(vx²+vy²+vz²) verificado analíticamente."""
    state = SimState()
    state.particles = make_particles(1)
    state.particles["vel"][0]  = [3.0, 4.0, 0.0]
    state.particles["pos"][0]  = [1.0, 1.0, 1.0]
    state.particles["mass"][0] = 2.0

    # Ek = ½ × 2 × (9 + 16 + 0) = 25.0
    d = compute_diagnostics(state)
    assert d.kinetic_energy == pytest.approx(25.0, abs=1e-10)


# ─────────────────────────────────────────────────────────────────
# Test 2: momento cero con velocidades opuestas
# ─────────────────────────────────────────────────────────────────
def test_diag_momento_cero():
    """P = 0 cuando dos partículas iguales tienen velocidades opuestas."""
    state = SimState()
    state.particles = make_particles(2)
    m = 0.5
    state.particles["mass"]    = m
    state.particles["vel"][0]  = [ 2.0, -1.0,  3.0]
    state.particles["vel"][1]  = [-2.0,  1.0, -3.0]
    state.particles["pos"][:] = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]

    d = compute_diagnostics(state)
    np.testing.assert_allclose(d.momentum, 0.0, atol=1e-12)


# ─────────────────────────────────────────────────────────────────
# Test 3: todas en reposo → Ek = 0, |P| = 0
# ─────────────────────────────────────────────────────────────────
def test_diag_reposo():
    """Partículas en reposo: Ek = 0 y momento = 0."""
    state = SimState()
    state.particles = make_particles(cfg.N_PARTICLES)
    state.particles["vel"] = 0.0
    state.particles["pos"] = 1.0

    d = compute_diagnostics(state)
    assert d.kinetic_energy      == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(d.momentum, 0.0, atol=1e-12)


# ─────────────────────────────────────────────────────────────────
# Test 4: Etot == Ek + Ep (consistencia)
# ─────────────────────────────────────────────────────────────────
def test_diag_energia_total_consistente():
    """total_energy == kinetic_energy + potential_energy."""
    state = SimState()
    n = 20
    state.particles = make_particles(n)
    state.particles["mass"]  = 0.05
    state.particles["vel"][:, 0] = np.arange(n) * 0.1
    state.particles["pos"][:, 0] = (np.arange(n) % cfg.NG) + 0.5

    # Potencial no trivial
    N = cfg.NG
    ix = np.arange(N)
    state.potential = 0.01 * np.sin(
        ix[:, None, None] * 0.1 +
        ix[None, :, None] * 0.17 +
        ix[None, None, :] * 0.23
    )

    d = compute_diagnostics(state)
    assert d.total_energy == pytest.approx(
        d.kinetic_energy + d.potential_energy, abs=1e-12
    )


# ─────────────────────────────────────────────────────────────────
# Test 5: Ek escala linealmente con masa
# Verifica que np.sum (equivalente a OMP reduction) acumula bien.
# ─────────────────────────────────────────────────────────────────
def test_diag_ek_escala_masa():
    """Duplicar la masa de todas las partículas duplica Ek y P."""
    n = 500
    rng = np.random.default_rng(0)

    # Estado base
    s1 = SimState()
    s1.particles = make_particles(n)
    s1.particles["vel"] = rng.standard_normal((n, 3))
    s1.particles["pos"][:, 0] = np.arange(n, dtype=float) % cfg.NG + 0.5
    s1.particles["pos"][:, 1] = 1.0
    s1.particles["pos"][:, 2] = 1.0
    m = 1.0 / n
    s1.particles["mass"] = m

    # Estado con masa doble
    s2 = SimState()
    s2.particles = make_particles(n)
    s2.particles["vel"]  = s1.particles["vel"].copy()
    s2.particles["pos"]  = s1.particles["pos"].copy()
    s2.particles["mass"] = 2.0 * m

    d1 = compute_diagnostics(s1)
    d2 = compute_diagnostics(s2)

    assert d2.kinetic_energy == pytest.approx(2.0 * d1.kinetic_energy, rel=1e-10)
    np.testing.assert_allclose(d2.momentum, 2.0 * d1.momentum, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────
# Test 6 (extra): cociente virial = |Ep| / (2·Ek)
# ─────────────────────────────────────────────────────────────────
def test_diag_virial_ratio():
    """virial_ratio == |Ep| / (2·Ek) verificado analíticamente."""
    state = SimState()
    state.particles = make_particles(1)
    state.particles["vel"][0]  = [1.0, 0.0, 0.0]
    state.particles["pos"][0]  = [1.0, 1.0, 1.0]
    state.particles["mass"][0] = 1.0

    # Asignar potencial conocido en la celda (1,1,1)
    state.potential[1, 1, 1] = -4.0   # Ep = ½ × 1 × (-4) = -2

    d = compute_diagnostics(state)

    # Ek = ½ × 1 × 1² = 0.5
    # Ep = ½ × 1 × (-4) = -2.0
    # virial = |-2| / (2 × 0.5) = 2.0
    assert d.kinetic_energy   == pytest.approx(0.5,  abs=1e-10)
    assert d.potential_energy == pytest.approx(-2.0, abs=1e-10)
    assert d.virial_ratio     == pytest.approx(2.0,  abs=1e-10)
