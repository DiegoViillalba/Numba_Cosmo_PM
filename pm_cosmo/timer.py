"""
timer.py
========
Medición de tiempos de pared por etapa de la simulación.

Equivalente de include/timer.hpp en la versión C++.

Usamos time.perf_counter() que es el equivalente Python de omp_get_wtime():
mide tiempo de pared real, no tiempo de CPU acumulado de todos los hilos.
Esto es esencial para medir correctamente el speedup en paralelo.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List


# ──────────────────────────────────────────────────────────────────
# ScopedTimer — context manager equivalente a ScopedTimer en C++
# ──────────────────────────────────────────────────────────────────
@contextmanager
def scoped_timer(label: str, print_result: bool = True):
    """
    Context manager que mide el tiempo de un bloque y lo imprime.

    Uso::

        with scoped_timer("CIC deposit"):
            cic_deposit(state)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        if print_result:
            print(f"[timer] {label:<24s} : {elapsed:.4f} s")


# ──────────────────────────────────────────────────────────────────
# StageTimer — acumula tiempos por etapa a lo largo del loop
# ──────────────────────────────────────────────────────────────────
class StageTimer:
    """
    Acumula tiempos por etapa durante múltiples pasos de simulación.

    Equivalente de StageTimer en timer.hpp (C++).

    Uso::

        timer = StageTimer()
        for step in range(N):
            timer.start("CIC")
            cic_deposit(state)
            timer.stop("CIC")
            ...
        timer.report(n_workers=4)
    """

    def __init__(self) -> None:
        self._starts: Dict[str, float] = {}
        self._totals: Dict[str, float] = {}
        self._order:  List[str]        = []

    def start(self, name: str) -> None:
        """Inicia el cronómetro para una etapa."""
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        """Detiene el cronómetro y acumula el tiempo."""
        dt = time.perf_counter() - self._starts[name]
        self._totals[name] = self._totals.get(name, 0.0) + dt
        if name not in self._order:
            self._order.append(name)

    def total(self, name: str) -> float:
        """Devuelve el tiempo total acumulado de una etapa."""
        return self._totals.get(name, 0.0)

    def grand_total(self) -> float:
        """Suma de todos los tiempos registrados."""
        return sum(self._totals.values())

    def report(self, n_workers: int = 1,
               serial: "StageTimer | None" = None) -> None:
        """
        Imprime la tabla de desglose de tiempos con speedup por etapa.

        Parameters
        ----------
        n_workers : número de workers/hilos usados (para la cabecera)
        serial    : StageTimer del run secuencial para calcular speedup
        """
        tot = self.grand_total()
        w = 22
        print()
        print("╔══════════════════════════════════════════════════════╗")
        print(f"║     Desglose de tiempos  |  workers = {n_workers:2d}            ║")
        print("╠══════════════════════════════════════════════════════╣")
        print(f"║ {'Etapa':<{w}}{'t (s)':>8}{'%':>7}{'speedup':>10} ║")
        print("╠══════════════════════════════════════════════════════╣")

        for name in self._order:
            t = self._totals[name]
            pct = 100.0 * t / tot if tot > 0 else 0.0
            sp  = (serial.total(name) / t
                   if serial and t > 0 and serial.total(name) > 0
                   else 1.0)
            print(f"║ {name:<{w}}{t:8.4f}{pct:6.1f}%{sp:9.2f}x ║")

        global_sp = (serial.grand_total() / tot
                     if serial and tot > 0 and serial.grand_total() > 0
                     else 1.0)
        eff = global_sp / max(n_workers, 1)

        print("╠══════════════════════════════════════════════════════╣")
        print(f"║ {'TOTAL':<{w}}{tot:8.4f}{'100%':>7}{global_sp:9.2f}x ║")
        print(f"║ {'Eficiencia':<{w+14}}{eff*100:.1f}%          ║")
        print("╚══════════════════════════════════════════════════════╝")
        print()
