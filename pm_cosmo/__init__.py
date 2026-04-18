"""
pm_cosmo
========
Simulador Particle-Mesh cosmológico en Python.

Módulos
-------
config        Parámetros globales (Ng, DT, N_STEPS…)
types         SimState, Diagnostics, PARTICLE_DTYPE
timer         StageTimer, scoped_timer
ic_reader     read_ic, generate_uniform_ic
cic           cic_deposit
poisson_fft   solve_poisson
gradient      compute_gradient
force_interp  interpolate_force
integrator    leapfrog_step, leapfrog_half_kick
diagnostics   compute_diagnostics, print_diagnostics
output        write_snapshot_ascii/binary, append_diagnostics_csv
simulation    Simulation, RunConfig
"""

from .config import NG, NP, N_PARTICLES, N_CELLS, DT, N_STEPS, G_GRAV
from .types import SimState, Diagnostics, PARTICLE_DTYPE
from .simulation import Simulation, RunConfig
