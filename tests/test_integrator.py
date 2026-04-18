"""
test_integrator.py
==================
Tests para el módulo integrator.py (Leap-Frog).

Propiedades verificadas — espejo de test_integrator.cpp:
    1. MRU: sin fuerza, x(t) = x0 + v0·t (módulo Ng).
    2. Periodicidad: partícula que cruza el borde reaparece al otro lado.
    3. Velocidad constante con a=0 tras 50 pasos.
    4. MUA: x = ½·a·t² exacto con half-kick inicial.
    5. (Extra Python) Leap-Frog conserva energía para oscilador armónico.
"""

import numpy as np
import pytest

from pm_cosmo.types import SimState, make_particles
from pm_cosmo.integrator import leapfrog_step, leapfrog_half_kick
from pm_cosmo import config as cfg


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _one_particle(x, y, z, vx=0.0, vy=0.0, vz=0.0) -> SimState:
    state = SimState()
    state.particles = make_particles(1)
    state.particles["pos"][0] = [x, y, z]
    state.particles["vel"][0] = [vx, vy, vz]
    return state


def _zero_accel(n: int = 1) -> np.ndarray:
    return np.zeros((n, 3), dtype=np.float64)


# ─────────────────────────────────────────────────────────────────
# Test 1: movimiento rectilíneo uniforme (a = 0)
# x(t) = (x0 + v0 · N · dt) mod Ng
# ─────────────────────────────────────────────────────────────────
def test_integrador_mru():
    """Sin fuerza, la posición avanza linealmente con wraparound."""
    dt = cfg.DT
    Ng = cfg.NG
    N_STEPS_T = 10

    state  = _one_particle(10.0, 20.0, 30.0, vx=1.0, vy=0.5, vz=-0.3)
    accel  = _zero_accel(1)

    x0, y0, z0 = 10.0, 20.0, 30.0
    vx, vy, vz  = 1.0, 0.5, -0.3

    for _ in range(N_STEPS_T):
        leapfrog_step(state, accel)

    expected_x = (x0 + vx * dt * N_STEPS_T) % Ng
    expected_y = (y0 + vy * dt * N_STEPS_T) % Ng
    expected_z = (z0 + vz * dt * N_STEPS_T) % Ng

    pos = state.particles["pos"][0]
    np.testing.assert_allclose(pos[0], expected_x, atol=1e-10)
    np.testing.assert_allclose(pos[1], expected_y, atol=1e-10)
    np.testing.assert_allclose(pos[2], expected_z, atol=1e-10)


# ─────────────────────────────────────────────────────────────────
# Test 2: condiciones de frontera periódicas
# Partícula a dt/2 del borde derecho con vel=1 → cruza en 1 paso.
# ─────────────────────────────────────────────────────────────────
def test_integrador_periodicidad():
    """Partícula que sale por x=Ng reaparece por x≈0."""
    dt = cfg.DT
    Ng = float(cfg.NG)

    state = _one_particle(Ng - dt * 0.5, Ng / 2.0, Ng / 2.0, vx=1.0)
    accel = _zero_accel(1)

    leapfrog_step(state, accel)

    x = state.particles["pos"][0, 0]
    assert 0.0 <= x < Ng, f"Posición fuera del dominio: x={x}"
    np.testing.assert_allclose(x, dt * 0.5, atol=1e-10)


# ─────────────────────────────────────────────────────────────────
# Test 3: velocidad constante con a = 0 tras 50 pasos
# ─────────────────────────────────────────────────────────────────
def test_integrador_vel_constante():
    """Velocidad no cambia cuando la aceleración es cero."""
    state = _one_particle(10.0, 10.0, 10.0, vx=2.5, vy=-1.3, vz=0.7)
    accel = _zero_accel(1)

    for _ in range(50):
        leapfrog_step(state, accel)

    vel = state.particles["vel"][0]
    np.testing.assert_allclose(vel[0],  2.5, atol=1e-12)
    np.testing.assert_allclose(vel[1], -1.3, atol=1e-12)
    np.testing.assert_allclose(vel[2],  0.7, atol=1e-12)


# ─────────────────────────────────────────────────────────────────
# Test 4: movimiento uniformemente acelerado (a = cte)
# El Leap-Frog es exacto para a constante: x = ½·a·t²
# ─────────────────────────────────────────────────────────────────
def test_integrador_mua():
    """x(t) = ½·a·t² exacto con Leap-Frog + half-kick inicial."""
    dt = cfg.DT
    ax = 0.5
    N_STEPS_T = 10

    state = _one_particle(0.0, 0.0, 0.0)   # v0 = 0, x0 = 0
    accel = np.array([[ax, 0.0, 0.0]], dtype=np.float64)

    # Half-kick para sincronizar velocidades a t = -dt/2
    leapfrog_half_kick(state, accel)

    for _ in range(N_STEPS_T):
        leapfrog_step(state, accel)

    t_final    = N_STEPS_T * dt
    x_expected = 0.5 * ax * t_final**2

    x = state.particles["pos"][0, 0]
    np.testing.assert_allclose(x, x_expected, atol=1e-10,
        err_msg=f"x={x:.6e}, esperado {x_expected:.6e}")


# ─────────────────────────────────────────────────────────────────
# Test 5: estado no cambia tras 0 pasos
# ─────────────────────────────────────────────────────────────────
def test_integrador_cero_pasos():
    """No ejecutar ningún paso no modifica el estado."""
    state = _one_particle(12.0, 34.0, 56.0, vx=1.0, vy=-2.0, vz=3.0)
    pos_before = state.particles["pos"].copy()
    vel_before = state.particles["vel"].copy()

    # No se llama a leapfrog_step
    np.testing.assert_array_equal(state.particles["pos"], pos_before)
    np.testing.assert_array_equal(state.particles["vel"], vel_before)


# ─────────────────────────────────────────────────────────────────
# Test 6 (extra Python): Leap-Frog para oscilador armónico
# Para F = -ω²·x la energía E = ½v² + ½ω²x² se conserva a
# orden O(dt²) por turno, con error acotado globalmente.
# ─────────────────────────────────────────────────────────────────
def test_integrador_conservacion_energia_oscilador():
    """
    Leap-Frog en oscilador armónico: la energía no deriva.

    El Leap-Frog es simpléctico: la energía oscila alrededor de E0
    pero NO crece ni decrece sistemáticamente con el tiempo.
    Verificamos que E_final esté dentro del 10% de E_inicial,
    sin tendencia monotónica.
    """
    omega  = 1.0
    dt     = 0.1
    N_PASOS = 1000   # ~16 períodos: suficiente para detectar deriva

    x = 1.0
    v = 0.0
    v -= (-omega**2 * x) * (dt / 2.0)   # half-kick

    energias = []
    for _ in range(N_PASOS):
        v += (-omega**2 * x) * dt
        x += v * dt
        energias.append(0.5 * v**2 + 0.5 * omega**2 * x**2)

    E0    = energias[0]
    E_fin = energias[-1]

    # 1. Sin deriva: E_final dentro del 10% de E0
    assert abs(E_fin - E0) / (abs(E0) + 1e-30) < 0.10, (
        f"Posible deriva: E0={E0:.6f}, E_fin={E_fin:.6f}"
    )

    # 2. Sin crecimiento monotónico: promedio de la segunda mitad ≈ primera mitad
    mitad = N_PASOS // 2
    E_media_1 = sum(energias[:mitad]) / mitad
    E_media_2 = sum(energias[mitad:]) / mitad
    assert abs(E_media_2 - E_media_1) / (abs(E_media_1) + 1e-30) < 0.01, (
        f"Deriva en la media de energía: "
        f"E_media_1={E_media_1:.6f}, E_media_2={E_media_2:.6f}"
    )
