#!/usr/bin/env python3
"""
run_bench.py
============
Benchmarking automático del simulador PM en Python.

Corre la simulación con distintos números de workers numba,
extrae tiempos por etapa y genera una tabla de speedup.

Uso::

    python bench/run_bench.py [--steps N] [--workers 1 2 4 8] [--ng 128]
    python bench/run_bench.py --steps 20 --workers 1 2 4 8 16

Salida: imprime tabla en stdout y guarda bench/results.csv.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Asegurar que pm_cosmo sea importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from pm_cosmo import config as cfg
from pm_cosmo.simulation import Simulation, RunConfig
from pm_cosmo.timer import StageTimer


def run_one(n_workers: int, n_steps: int, ng: int) -> dict:
    """
    Ejecuta una corrida completa y devuelve un dict con tiempos por etapa.

    Parameters
    ----------
    n_workers : número de workers de numba
    n_steps   : pasos de simulación
    ng        : tamaño de malla
    """
    # Configurar NG dinámicamente
    cfg.NG          = ng
    cfg.NP          = ng
    cfg.N_PARTICLES = ng ** 3
    cfg.N_CELLS     = ng ** 3
    cfg.CELL_SIZE   = cfg.L_BOX / ng

    # Limpiar caché de k² para que se recalcule con el NG nuevo
    import pm_cosmo.poisson_fft as pf
    pf._K2 = None

    run_cfg = RunConfig(
        ic_file    = "",
        n_workers  = n_workers,
        n_steps    = n_steps,
        output_dir = f"/tmp/pm_bench_{ng}_{n_workers}w",
        use_ascii  = False,
        verbose    = False,
    )

    sim = Simulation(run_cfg)
    sim.init()

    t0    = time.perf_counter()
    timer = sim.run()
    total = time.perf_counter() - t0

    return {
        "workers":  n_workers,
        "total_s":  round(total, 4),
        "cic_s":    round(timer.total("1. CIC deposit"),   4),
        "fft_s":    round(timer.total("2. Poisson FFT"),   4),
        "grad_s":   round(timer.total("3. Gradiente"),     4),
        "interp_s": round(timer.total("4. Force interp"),  4),
        "integr_s": round(timer.total("5. Integrador"),    4),
    }


def print_table(rows: list[dict], t_serial: float) -> None:
    """Imprime tabla formateada de speedup."""
    print()
    print(f"{'workers':<10} {'tiempo(s)':<12} {'speedup':<10} {'eficiencia':<12}")
    print("-" * 46)
    for r in rows:
        sp  = t_serial / r["total_s"] if r["total_s"] > 0 else 1.0
        eff = sp / r["workers"]
        print(f"{r['workers']:<10} {r['total_s']:<12.4f} {sp:<10.2f} {eff*100:<12.1f}%")
    print()


def save_csv(rows: list[dict], output: str) -> None:
    """Guarda resultados en CSV."""
    fields = ["workers", "total_s", "cic_s", "fft_s",
              "grad_s", "interp_s", "integr_s"]
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Resultados guardados en: {output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark pm_cosmo Python")
    p.add_argument("--steps",   type=int, default=20)
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--ng",      type=int, default=128)
    p.add_argument("--output",  default="bench/results_py.csv")
    args = p.parse_args()

    print(f"\n=== Benchmark pm_cosmo Python ===")
    print(f"  Ng={args.ng}, pasos={args.steps}, workers={args.workers}\n")

    rows     = []
    t_serial = None

    for w in args.workers:
        print(f"--- Corriendo con {w} worker(s) ---")
        row = run_one(w, args.steps, args.ng)
        rows.append(row)
        if t_serial is None:
            t_serial = row["total_s"]
        print(f"    Total: {row['total_s']:.4f}s")
        print()

    print_table(rows, t_serial)
    Path(args.output).parent.mkdir(exist_ok=True)
    save_csv(rows, args.output)


if __name__ == "__main__":
    main()
