# Cloudy C25 installation and verification

Status: **installed, compiled, and verified end to end** on 2026-07-17.

## Installed build

- Source revision: `cf48c496bacaa0269b5d0623b92f8aa7b3e0aa26`
- Cloudy runtime identifier: `Cloudy (master, cf48c49)`, dated `25Aug22`
- Executable: `work/cloudy/source/cloudy.exe`
- Platform: native macOS arm64 Mach-O
- Compiler: Apple clang 17.0.0, C++17 optimized build
- Executable size: 5.5 MB
- Installation footprint: 4.6 GB, including 3.2 GB of atomic data
- SHA-256: `8645c443518db0d5f21aa12f4ea23b0f78c8bc92ec30199639c2a9bb449ddeeb`

The host advertises the unsupported locale `C.UTF-8`. Cloudy's Perl-based
dependency generator therefore fails under the inherited locale. The build
was completed with a supported deterministic locale:

```bash
cd work/cloudy/source
env LC_ALL=C LANG=C make -j2
```

This locale override is also applied automatically by `cloudy_bridge.py` at
runtime.

## End-to-end acceptance test

The test deck in `cloudy_smoke_test/model.in` runs an AGN-photoionized slab
with log n_H = 4, log U = -2, log N_H = 20, and T = 10,000 K. It asks Cloudy
for H-alpha and H-beta volume emissivities using the C25 multiline line-list
syntax and iterates to convergence.

Acceptance result:

- process exit code: 0
- Cloudy status: `Cloudy exited OK`
- zones: 35
- iterations: 3
- convergence cautions: 0 in the final run
- final-zone H-alpha emissivity: `3.7252e-17 erg cm^-3 s^-1`
- final-zone H-beta emissivity: `1.2880e-17 erg cm^-3 s^-1`
- final-zone Balmer decrement H-alpha/H-beta: `2.892236`

The full input, main output, and line table are retained in
`cloudy_smoke_test/`.

## Model integration

`cloudy_bridge.py` now:

1. finds the verified local build automatically;
2. permits an explicit executable or `CLOUDY_EXE` override;
3. generates valid C25 multiline H-alpha/H-beta requests;
4. requests iteration to convergence;
5. forces the supported `C` locale; and
6. fails explicitly on a missing executable, Cloudy error, or timeout.

The bridge and all existing semi-analytic lifecycle tests pass:
23 tests total, including Cloudy syntax, normalization, emulator, and
lifecycle-integration tests.

## Reproduce the smoke test through the bridge

From `outputs/semi_analytic_lifecycle/cloudy_smoke_test`:

```bash
PYTHONPATH=.. python3 -c 'from cloudy_bridge import run_cloudy; run_cloudy(".")'
```

For a different installation:

```bash
export CLOUDY_EXE=/absolute/path/to/cloudy.exe
```

This verifies installation and the H-line interface. It does not by itself
validate the physical assumptions of the eventual LRD grid; density,
ionization parameter, column, metallicity, dust, geometry, incident SED, and
line-to-observable integration still require a designed grid and data-level
calibration.
