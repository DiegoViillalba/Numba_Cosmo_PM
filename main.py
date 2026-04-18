#!/usr/bin/env python3
"""
main.py
=======
Punto de entrada del simulador PM cosmológico en Python.

Uso::

    python main.py [opciones] [ic_file]

Opciones::

    -t, --threads N   Workers de numba (default: 1)
    -s, --steps N     Número de pasos  (default: N_STEPS de config.py)
    -o, --output DIR  Directorio de salida (default: data)
    -b, --binary      Snapshots en formato binario
    -v, --verbose     Imprimir diagnósticos en cada paso
    -g, --generate    Generar IC sintéticas (ignorar ic_file)
    --ng N            Sobreescribir NG (tamaño de malla)

Ejemplos::

    python main.py -g -t 4 -s 20 -v
    python main.py ic/ic_128.dat -t 8 -s 100 -o resultados/
    python main.py -g --ng 128 -s 10
"""

import argparse
import sys

from pm_cosmo.simulation import Simulation, RunConfig
from pm_cosmo import config as cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulador Particle-Mesh Cosmológico — UNAM TSCAD 2026-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("ic_file",       nargs="?", default="",
                   help="Archivo de condiciones iniciales (ASCII)")
    p.add_argument("-t", "--threads", type=int, default=1,
                   help="Workers de numba (default: 1)")
    p.add_argument("-s", "--steps",   type=int, default=0,
                   help="Número de pasos (0 = usar N_STEPS de config)")
    p.add_argument("-o", "--output",  default="data",
                   help="Directorio de salida (default: data)")
    p.add_argument("-b", "--binary",  action="store_true",
                   help="Snapshots binarios")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Diagnósticos en cada paso")
    p.add_argument("-g", "--generate", action="store_true",
                   help="Generar IC sintéticas (no requiere archivo)")
    p.add_argument("--ng", type=int, default=None,
                   help="Sobreescribir NG (tamaño de malla)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Sobreescribir NG si se pasó como argumento ─────────────────
    if args.ng is not None:
        import pm_cosmo.config as c
        c.NG         = args.ng
        c.NP         = args.ng
        c.N_PARTICLES = args.ng ** 3
        c.N_CELLS    = args.ng ** 3
        c.CELL_SIZE  = c.L_BOX / args.ng

    # ── Banner ─────────────────────────────────────────────────────
    print()
    print("╔════════════════════════════════════════════════════╗")
    print("║    Simulador Particle-Mesh Cosmológico — UNAM      ║")
    print("║    TSCAD 2026-2  |  Física Biomédica  (Python)     ║")
    print("╚════════════════════════════════════════════════════╝")
    print()

    # ── Configuración ──────────────────────────────────────────────
    run_cfg = RunConfig(
        ic_file    = "" if args.generate else args.ic_file,
        n_workers  = args.threads,
        n_steps    = args.steps,
        output_dir = args.output,
        use_ascii  = not args.binary,
        verbose    = args.verbose,
    )

    # ── Ejecutar ───────────────────────────────────────────────────
    try:
        sim = Simulation(run_cfg)
        sim.init()
        sim.run()
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[main] Simulación completada. Resultados en: {args.output}/\n")


if __name__ == "__main__":
    main()
