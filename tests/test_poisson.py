"""
test_poisson.py
===============
Tests para el módulo poisson_fft.py.

Propiedades verificadas — espejo de test_poisson.cpp:
    1. Modo DC nulo: densidad constante → potencial ≈ 0.
    2. Forma coseno: perfil de φ correlaciona > 0.99 con coseno.
    3. No NaN/Inf para densidad arbitraria.
    4. Linealidad: φ(α·ρ) = α·φ(ρ).
    5. (Extra Python) Simetría: φ debe ser real (irfftn garantiza esto).
"""

import numpy as np
import pytest

from pm_cosmo.types import SimState
from pm_cosmo.poisson_fft import solve_poisson, _get_k2
from pm_cosmo import config as cfg


# ─────────────────────────────────────────────────────────────────
# Test 1: modo DC → potencial cero
# ─────────────────────────────────────────────────────────────────
def test_poisson_modo_dc_cero():
    """Densidad uniforme (solo modo k=0) → potencial ≈ 0 en todas partes."""
    state = SimState()
    state.density[:] = 1.0

    solve_poisson(state)

    np.testing.assert_allclose(
        state.potential, 0.0, atol=1e-10,
        err_msg=f"Potencial máximo: {np.abs(state.potential).max():.3e}",
    )


# ─────────────────────────────────────────────────────────────────
# Test 2: perfil coseno
# ρ(ix) = 1 + cos(2π·ix/N) → φ debe ser proporcional a cos(2π·ix/N).
# ─────────────────────────────────────────────────────────────────
def test_poisson_forma_coseno():
    """El solver reproduce la forma coseno del modo fundamental."""
    N = cfg.NG
    state = SimState()

    ix = np.arange(N)
    cos_wave = np.cos(2.0 * np.pi * ix / N)  # (N,)

    # Densidad: 1 + cos en el eje x, constante en y, z
    state.density = 1.0 + cos_wave[:, None, None] * np.ones((1, N, N))

    solve_poisson(state)

    # Perfil promediado sobre y, z
    phi_profile = state.potential.mean(axis=(1, 2))  # (N,)

    # Correlación con coseno normalizada
    num     = float(np.dot(phi_profile, cos_wave))
    den_phi = float(np.dot(phi_profile, phi_profile))
    # Norma del coseno discreto: sqrt(N/2)
    corr = abs(num) / (np.sqrt(den_phi) * np.sqrt(N / 2.0) + 1e-30)

    assert corr > 0.99, (
        f"Correlación con coseno = {corr:.4f} < 0.99; "
        "el solver no reproduce la forma correcta."
    )


# ─────────────────────────────────────────────────────────────────
# Test 3: no NaN / Inf
# ─────────────────────────────────────────────────────────────────
def test_poisson_no_nan_inf():
    """El solver nunca produce NaN o Inf para densidad arbitraria."""
    state = SimState()
    N = cfg.NG
    ix = np.arange(N)
    state.density = 1.0 + 0.3 * np.sin(
        ix[:, None, None] * 0.007 +
        ix[None, :, None] * 0.013 +
        ix[None, None, :] * 0.019
    )

    solve_poisson(state)

    assert np.all(np.isfinite(state.potential)), (
        "El potencial contiene NaN o Inf"
    )


# ─────────────────────────────────────────────────────────────────
# Test 4: linealidad φ(α·ρ) = α·φ(ρ)
# ─────────────────────────────────────────────────────────────────
def test_poisson_linealidad():
    """φ(α·ρ) == α·φ(ρ) para cualquier escalar α."""
    alpha = 3.7
    N = cfg.NG

    # Run 1: densidad base
    s1 = SimState()
    s1.density = 1.0 + 0.5 * np.cos(
        np.arange(N)[:, None, None] * 0.05
    ) * np.ones((1, N, N))
    solve_poisson(s1)

    # Run 2: densidad escalada por alpha
    s2 = SimState()
    s2.density = alpha * s1.density.copy()
    solve_poisson(s2)

    np.testing.assert_allclose(
        s2.potential, alpha * s1.potential,
        atol=1e-8,
        err_msg="Violación de linealidad: φ(α·ρ) ≠ α·φ(ρ)",
    )


# ─────────────────────────────────────────────────────────────────
# Test 5 (extra Python): el potencial es siempre real
# irfftn devuelve un array float64 real; verificamos que no tiene
# parte imaginaria residual (garantía de la simetría hermítica).
# ─────────────────────────────────────────────────────────────────
def test_poisson_potencial_real():
    """El potencial resultante es real (array float64)."""
    state = SimState()
    state.density = 1.0 + 0.2 * np.random.default_rng(0).standard_normal(
        (cfg.NG, cfg.NG, cfg.NG)
    )
    solve_poisson(state)

    assert state.potential.dtype == np.float64, (
        f"dtype del potencial: {state.potential.dtype}, esperado float64"
    )
    assert state.potential.shape == (cfg.NG, cfg.NG, cfg.NG), (
        f"Shape del potencial: {state.potential.shape}"
    )


# ─────────────────────────────────────────────────────────────────
# Test 6: la malla k² precalculada tiene el modo DC = 0
# ─────────────────────────────────────────────────────────────────
def test_k2_grid_dc_es_cero():
    """El primer elemento de la malla k² (modo DC) debe ser 0."""
    k2 = _get_k2()
    assert k2[0, 0, 0] == pytest.approx(0.0, abs=1e-30), (
        f"k²[0,0,0] = {k2[0,0,0]}, esperado 0"
    )
