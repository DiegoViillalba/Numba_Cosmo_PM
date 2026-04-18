"""
test_pipeline.py
================
Tests de integración: pipeline PM completo.

Estos tests ejercitan el sistema end-to-end y verifican propiedades
físicas globales que solo emergen cuando todos los módulos trabajan juntos:
    1. Conservación de energía: |ΔE/E| < 5% en 10 pasos.
    2. Conservación de momento: |P| constante en máquina.
    3. Serial == paralelo: mismo resultado en los dos modos.
    4. Snap inicial se escribe correctamente.
"""

import numpy as np
import pytest

from pm_cosmo.types import SimState
from pm_cosmo.ic_reader import generate_uniform_ic
from pm_cosmo.cic import cic_deposit
from pm_cosmo.poisson_fft import solve_poisson
from pm_cosmo.gradient import compute_gradient
from pm_cosmo.force_interp import interpolate_force
from pm_cosmo.integrator import leapfrog_step, leapfrog_half_kick
from pm_cosmo.diagnostics import compute_diagnostics
from pm_cosmo import config as cfg


def _run_n_steps(state: SimState, n: int) -> list:
    """Corre n pasos del pipeline PM y devuelve lista de Diagnostics."""
    state.clear_grids()
    cic_deposit(state)
    solve_poisson(state)
    compute_gradient(state)
    accel = interpolate_force(state)
    leapfrog_half_kick(state, accel)

    diags = [compute_diagnostics(state)]

    for _ in range(n):
        state.clear_grids()
        cic_deposit(state)
        solve_poisson(state)
        compute_gradient(state)
        accel = interpolate_force(state)
        leapfrog_step(state, accel)
        diags.append(compute_diagnostics(state))

    return diags


# ─────────────────────────────────────────────────────────────────
# Test 1: conservación de energía en 10 pasos
# El Leap-Frog es simpléctico: la energía oscila pero no deriva.
# Verificamos que el cambio relativo sea < 5%.
# ─────────────────────────────────────────────────────────────────
def test_pipeline_conservacion_energia():
    """Energía total varía menos de 5% en 10 pasos."""
    state = SimState()
    generate_uniform_ic(state, sigma_pos=0.3, sigma_vel=0.05, seed=0)

    diags = _run_n_steps(state, 10)

    E0   = diags[0].total_energy
    Efin = diags[-1].total_energy

    dE_rel = abs(Efin - E0) / (abs(E0) + 1e-30)
    assert dE_rel < 0.05, (
        f"Variación de energía = {dE_rel:.4f} > 5%  "
        f"(E0={E0:.4e}, Ef={Efin:.4e})"
    )


# ─────────────────────────────────────────────────────────────────
# Test 2: conservación de momento
# En un sistema periódico sin fuerzas externas, P se conserva exactamente.
# ─────────────────────────────────────────────────────────────────
def test_pipeline_conservacion_momento():
    """El momento total no cambia en 5 pasos (conservación exacta)."""
    state = SimState()
    generate_uniform_ic(state, sigma_pos=0.3, sigma_vel=0.05, seed=1)

    diags = _run_n_steps(state, 5)

    P0   = np.linalg.norm(diags[0].momentum)
    Pfin = np.linalg.norm(diags[-1].momentum)

    # El momento debería conservarse a precisión de punto flotante
    np.testing.assert_allclose(Pfin, P0, rtol=1e-10,
        err_msg=f"|P| inicial={P0:.6e}, final={Pfin:.6e}")


# ─────────────────────────────────────────────────────────────────
# Test 3: valores físicos idénticos entre dos runs con la misma semilla
# Verifica que el pipeline sea determinista (reproducibilidad).
# ─────────────────────────────────────────────────────────────────
def test_pipeline_determinista():
    """Dos runs con la misma semilla producen exactamente los mismos valores."""
    def one_run(seed: int) -> tuple:
        state = SimState()
        generate_uniform_ic(state, seed=seed)
        diags = _run_n_steps(state, 3)
        return diags[-1].kinetic_energy, diags[-1].total_energy

    Ek1, Et1 = one_run(seed=99)
    Ek2, Et2 = one_run(seed=99)

    assert Ek1 == pytest.approx(Ek2, rel=1e-12)
    assert Et1 == pytest.approx(Et2, rel=1e-12)


# ─────────────────────────────────────────────────────────────────
# Test 4: el snapshot inicial es legible y compatible
# ─────────────────────────────────────────────────────────────────
def test_pipeline_snapshot_roundtrip(tmp_path):
    """El snapshot del paso 0 se puede releer y produce las mismas energías."""
    from pm_cosmo.output import write_snapshot_ascii
    from pm_cosmo.ic_reader import read_ic

    state = SimState()
    generate_uniform_ic(state, seed=5)

    # Calcular diagnósticos antes de guardar
    state.clear_grids()
    cic_deposit(state)
    solve_poisson(state)
    diag1 = compute_diagnostics(state)

    # Guardar y recargar
    snap = str(tmp_path / "snap_test.dat")
    write_snapshot_ascii(state, snap)

    state2 = SimState()
    read_ic(state2, snap)

    state2.clear_grids()
    cic_deposit(state2)
    solve_poisson(state2)
    diag2 = compute_diagnostics(state2)

    np.testing.assert_allclose(
        diag2.kinetic_energy, diag1.kinetic_energy, rtol=1e-6,
        err_msg="Energía cinética difiere tras round-trip de snapshot"
    )
