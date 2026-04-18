"""
conftest.py
===========
Fixtures compartidas para la suite de tests de pm_cosmo.

pytest las descubre automáticamente en este archivo y las inyecta
en los tests que las soliciten como parámetro.
"""

import sys
import os
# Asegurar que el paquete pm_cosmo sea importable desde tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from pm_cosmo.types import SimState, make_particles
from pm_cosmo import config as cfg


@pytest.fixture
def empty_state() -> SimState:
    """SimState vacío (partículas en cero, mallas en cero)."""
    return SimState()


@pytest.fixture
def single_particle_state() -> SimState:
    """SimState con una sola partícula de masa=1 en (5, 5, 5)."""
    state = SimState()
    state.particles = make_particles(1)
    state.particles["pos"][0] = [5.0, 5.0, 5.0]
    state.particles["vel"][0] = [0.0, 0.0, 0.0]
    state.particles["mass"][0] = 1.0
    return state


@pytest.fixture
def uniform_grid_state() -> SimState:
    """
    SimState con partículas en grilla regular perfecta.
    Cada partícula está en el centro exacto de su celda.
    """
    from pm_cosmo.ic_reader import generate_uniform_ic
    state = SimState()
    generate_uniform_ic(state, sigma_pos=0.0, sigma_vel=0.0)
    return state


@pytest.fixture
def perturbed_state() -> SimState:
    """SimState con IC sintéticas perturbadas (para tests de integración)."""
    from pm_cosmo.ic_reader import generate_uniform_ic
    state = SimState()
    generate_uniform_ic(state, sigma_pos=0.5, sigma_vel=0.1, seed=42)
    return state
