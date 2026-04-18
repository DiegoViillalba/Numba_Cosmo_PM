# pm\_cosmo — Python

Implementación Python del simulador Particle-Mesh cosmológico.  
Equivalente módulo a módulo de la versión C++/OpenMP.

---

## Dependencias

```bash
pip install -r requirements.txt
# numpy, scipy, numba, pytest
```

---

## Uso rápido

```bash
# IC sintéticas, 20 pasos, verbose
python main.py -g -s 20 -v

# Con archivo de CI externo, 4 workers numba
python main.py ic/ic_128.dat -t 4 -s 100 -o resultados/

# Opciones
#   -t N   Workers de numba (default: 1)
#   -s N   Pasos de simulación
#   -o DIR Directorio de salida
#   -g     Generar IC sintéticas
#   -b     Snapshots binarios
#   -v     Verbose
#  --ng N  Tamaño de malla (128 o 256)
```

---

## Tests

```bash
python -m pytest tests/ -v          # todos los tests (38)
python -m pytest tests/test_cic.py  # solo CIC
python -m pytest tests/ -k pipeline # solo tests de integración
```

---

## Benchmark

```bash
python bench/run_bench.py --steps 20 --workers 1 2 4 --ng 128
```

---

## Estructura

```
pm_cosmo_py/
├── pm_cosmo/
│   ├── config.py        Parámetros globales (NG, DT, N_STEPS…)
│   ├── types.py         SimState, Diagnostics, PARTICLE_DTYPE
│   ├── timer.py         StageTimer (perf_counter)
│   ├── ic_reader.py     read_ic, generate_uniform_ic
│   ├── cic.py           cic_deposit (numba prange + acumuladores locales)
│   ├── poisson_fft.py   solve_poisson (numpy.fft.rfftn)
│   ├── gradient.py      compute_gradient (numba prange)
│   ├── force_interp.py  interpolate_force (numba prange)
│   ├── integrator.py    leapfrog_step, leapfrog_half_kick (numba prange)
│   ├── diagnostics.py   compute_diagnostics (numpy vectorizado)
│   ├── output.py        write_snapshot_ascii/binary, append_diagnostics_csv
│   └── simulation.py    Simulation, RunConfig
├── tests/
│   ├── conftest.py          Fixtures pytest compartidas
│   ├── test_cic.py          5 tests del depósito CIC
│   ├── test_poisson.py      6 tests del solucionador Poisson
│   ├── test_integrator.py   6 tests del integrador Leap-Frog
│   ├── test_diagnostics.py  6 tests de energías y momento
│   ├── test_gradient.py     4 tests del gradiente
│   ├── test_ic_reader.py    6 tests del lector de IC
│   └── test_pipeline.py     4 tests de integración end-to-end
├── bench/
│   └── run_bench.py     Benchmarking con tabla de speedup
├── main.py              Punto de entrada (argparse)
└── requirements.txt
```

---

## Paralelismo por módulo

| Módulo | Mecanismo | Equivalente C++ |
|---|---|---|
| `cic.py` | numba `prange` + acumuladores locales | `#pragma omp parallel for` + acumuladores locales |
| `poisson_fft.py` | numpy `rfftn` (interno) | FFTW3 |
| `gradient.py` | numba `prange` | `#pragma omp parallel for collapse(3)` |
| `force_interp.py` | numba `prange` | `#pragma omp parallel for schedule(static)` |
| `integrator.py` | numba `prange` | `#pragma omp parallel for schedule(static)` |
| `diagnostics.py` | numpy `sum` / `dot` | `#pragma omp parallel for reduction(+:Ek)` |
