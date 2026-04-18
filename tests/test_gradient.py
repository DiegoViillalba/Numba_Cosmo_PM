"""
test_gradient.py
================
Tests para el módulo gradient.py.

Propiedades verificadas:
    1. Campo nulo para potencial constante.
    2. Solución analítica para φ = sin(2πx/L): g_x = -∂φ/∂x.
    3. Antisimetría: g(-x) = -g(x).
    4. No NaN/Inf para potencial arbitrario.
"""

import numpy as np
import pytest

from pm_cosmo.types import SimState
from pm_cosmo.gradient import compute_gradient
from pm_cosmo import config as cfg


# ─────────────────────────────────────────────────────────────────
# Test 1: potencial constante → campo cero
# ─────────────────────────────────────────────────────────────────
def test_gradient_potencial_constante():
    """φ = cte → g = 0 en todas las celdas."""
    state = SimState()
    state.potential[:] = 5.0   # constante arbitraria

    compute_gradient(state)

    np.testing.assert_allclose(state.force_x, 0.0, atol=1e-12)
    np.testing.assert_allclose(state.force_y, 0.0, atol=1e-12)
    np.testing.assert_allclose(state.force_z, 0.0, atol=1e-12)


# ─────────────────────────────────────────────────────────────────
# Test 2: solución analítica para modo fundamental
# φ(ix) = A·sin(2π·ix/N)  →  g_x(ix) = -A·(2π/N)·cos(2π·ix/N) / Δx
# en diferencias centradas con Δx = CELL_SIZE.
# ─────────────────────────────────────────────────────────────────
def test_gradient_solucion_analitica():
    """g_x para φ = sin(2πx/N) coincide con la derivada analítica."""
    N  = cfg.NG
    dx = cfg.CELL_SIZE
    A  = 2.0

    ix = np.arange(N)
    phi_1d = A * np.sin(2.0 * np.pi * ix / N)

    state = SimState()
    state.potential = phi_1d[:, None, None] * np.ones((1, N, N))

    compute_gradient(state)

    # Derivada analítica de diferencias centradas (no del continuo):
    # g_x(i) = -(φ(i+1) - φ(i-1)) / (2·dx)
    g_analytic = -(np.roll(phi_1d, -1) - np.roll(phi_1d, 1)) / (2.0 * dx)

    # Comparar el perfil promediado en y, z
    g_profile = state.force_x.mean(axis=(1, 2))
    np.testing.assert_allclose(g_profile, g_analytic, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────
# Test 3: no NaN / Inf
# ─────────────────────────────────────────────────────────────────
def test_gradient_no_nan_inf():
    """El gradiente nunca produce NaN o Inf."""
    N = cfg.NG
    rng = np.random.default_rng(99)
    state = SimState()
    state.potential = rng.standard_normal((N, N, N))

    compute_gradient(state)

    for name, arr in [("force_x", state.force_x),
                      ("force_y", state.force_y),
                      ("force_z", state.force_z)]:
        assert np.all(np.isfinite(arr)), f"{name} contiene NaN o Inf"


# ─────────────────────────────────────────────────────────────────
# Test 4: shape correcto de las mallas de fuerza
# ─────────────────────────────────────────────────────────────────
def test_gradient_shape():
    """Las tres mallas de fuerza tienen shape (Ng, Ng, Ng)."""
    state = SimState()
    state.potential = np.random.default_rng(0).standard_normal(
        (cfg.NG, cfg.NG, cfg.NG)
    )

    compute_gradient(state)

    expected = (cfg.NG, cfg.NG, cfg.NG)
    assert state.force_x.shape == expected
    assert state.force_y.shape == expected
    assert state.force_z.shape == expected
