"""
simulation.py
=============
Orquestador del loop principal de la simulación PM.

Equivalente de src/simulation.cpp en la versión C++.

Pipeline de un paso de tiempo:
    1. cic_deposit        partículas → ρ(x)
    2. solve_poisson      ρ(x)       → φ(x)
    3. compute_gradient   φ(x)       → g(x)
    4. interpolate_force  g(x)       → a_i
    5. leapfrog_step      (x,v)      → (x',v')
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .types import SimState
from .timer import StageTimer
from .ic_reader import read_ic, generate_uniform_ic
from .cic import cic_deposit
from .poisson_fft import solve_poisson
from .gradient import compute_gradient
from .force_interp import interpolate_force
from .integrator import leapfrog_step, leapfrog_half_kick
from .diagnostics import compute_diagnostics, print_diagnostics
from .output import (write_snapshot_ascii, write_snapshot_binary,write_snapshot_hdf5_gadget,
                     append_diagnostics_csv)
from . import config as cfg


# ──────────────────────────────────────────────────────────────────
# RunConfig — equivalente de struct RunConfig en C++
# ──────────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    """Parámetros de ejecución pasados desde la línea de comandos."""
    ic_file:    str  = ""       # archivo de condiciones iniciales (vacío = generar)
    n_workers:  int  = 1        # workers para numba/multiprocessing
    n_steps:    int  = 0        # 0 = usar cfg.N_STEPS
    output_dir: str  = "data"   # carpeta de snapshots y CSV
    use_ascii:  bool = True     # True = ASCII, False = binario
    verbose:    bool = False    # imprimir diagnósticos en cada paso


# ──────────────────────────────────────────────────────────────────
# Simulation
# ──────────────────────────────────────────────────────────────────
class Simulation:
    """
    Encapsula el estado y el loop principal del simulador PM.

    Equivalente de class Simulation en simulation.hpp/cpp de C++.
    """

    def __init__(self, cfg_run: RunConfig) -> None:
        self.cfg    = cfg_run
        self.state  = SimState()
        self.timer  = StageTimer()

        # Configurar número de workers de numba
        if cfg_run.n_workers > 1:
            os.environ["NUMBA_NUM_THREADS"] = str(cfg_run.n_workers)

        Path(cfg_run.output_dir).mkdir(parents=True, exist_ok=True)

        try:
            import numba
            mode = f"numba {numba.__version__} ({cfg_run.n_workers} threads)"
        except ImportError:
            mode = "numpy (sin numba)"

        print(f"[sim] Backend de paralelismo: {mode}")

    def init(self) -> None:
        """Carga o genera las CI y calcula la fuerza inicial (t=0)."""
        n_steps = self.cfg.n_steps or cfg.N_STEPS

        print("\n[sim] Inicializando simulación...")
        print(f"[sim] Malla: {cfg.NG}^3 = {cfg.N_CELLS} celdas")
        print(f"[sim] Partículas: {cfg.N_PARTICLES}")
        print(f"[sim] Pasos: {n_steps}\n")

        # ── Condiciones iniciales ──────────────────────────────────
        if self.cfg.ic_file:
            read_ic(self.state, self.cfg.ic_file)
        else:
            print("[sim] No se proporcionó IC → generando grilla uniforme")
            generate_uniform_ic(self.state)

        # ── Fuerza inicial para half-kick ──────────────────────────
        print("[sim] Calculando fuerza inicial...")
        self.state.clear_grids()
        cic_deposit(self.state)
        solve_poisson(self.state)
        compute_gradient(self.state)
        accel = interpolate_force(self.state)
        leapfrog_half_kick(self.state, accel)

        # ── Diagnóstico en t=0 ────────────────────────────────────
        diag = compute_diagnostics(self.state)
        if self.cfg.verbose:
            print_diagnostics(0, 0.0, diag)

        diag_file = str(Path(self.cfg.output_dir) / "diagnostics.csv")
        append_diagnostics_csv(diag_file, 0, 0.0,
                               diag.kinetic_energy,
                               diag.potential_energy,
                               diag.total_energy)
        self._write_snapshot(0)
        print("[sim] Inicialización completa.\n")

    def run(self) -> StageTimer:
        """
        Ejecuta el loop principal y devuelve el StageTimer.

        Returns
        -------
        StageTimer con los tiempos acumulados por etapa.
        """
        n_steps   = self.cfg.n_steps or cfg.N_STEPS
        diag_file = str(Path(self.cfg.output_dir) / "diagnostics.csv")

        print("╔══════════════════════════════════════════════╗")
        print("║         Iniciando loop de simulación         ║")
        print("╚══════════════════════════════════════════════╝\n")

        t_run_start = time.perf_counter()

        for s in range(1, n_steps + 1):
            self.state.step = s
            self._single_step()

            # Diagnósticos cada 10 pasos o si verbose
            if self.cfg.verbose or s % 10 == 0:
                diag = compute_diagnostics(self.state)
                if self.cfg.verbose:
                    print_diagnostics(s, self.state.time, diag)
                append_diagnostics_csv(diag_file, s, self.state.time,
                                       diag.kinetic_energy,
                                       diag.potential_energy,
                                       diag.total_energy)

            if s>200 and s % cfg.SNAP_INTERVAL == 0:
                self._write_snapshot(s)

            # Barra de progreso
            if s % max(n_steps // 10, 1) == 0 or s == n_steps:
                pct     = int(100 * s / n_steps)
                elapsed = time.perf_counter() - t_run_start
                est     = elapsed * (n_steps - s) / s if s > 0 else 0
                print(f"  [{pct:3d}%] paso {s:5d}/{n_steps}"
                      f"  transcurrido: {elapsed:.1f}s"
                      f"  restante: ~{est:.1f}s")

        self.timer.report(self.cfg.n_workers)
        return self.timer

    # ── Métodos privados ───────────────────────────────────────────

    def _single_step(self) -> None:
        """Pipeline PM completo para un paso de tiempo."""
        self.state.clear_grids()

        self.timer.start("1. CIC deposit")
        cic_deposit(self.state)
        self.timer.stop("1. CIC deposit")

        self.timer.start("2. Poisson FFT")
        solve_poisson(self.state)
        self.timer.stop("2. Poisson FFT")

        self.timer.start("3. Gradiente")
        compute_gradient(self.state)
        self.timer.stop("3. Gradiente")

        self.timer.start("4. Force interp")
        accel = interpolate_force(self.state)
        self.timer.stop("4. Force interp")

        self.timer.start("5. Integrador")
        leapfrog_step(self.state, accel)
        self.timer.stop("5. Integrador")

    def _write_snapshot(self, step: int) -> None:
        """Escribe snapshot con el formato y nombre configurados."""
        name = str(Path(self.cfg.output_dir) / f"snap_{step:04d}")
        if self.cfg.use_ascii:
            write_snapshot_ascii(self.state, name + ".dat")
        else:
            # write_snapshot_binary(self.state, name + ".bin")
            write_snapshot_hdf5_gadget(self.state, name + ".hdf5")
