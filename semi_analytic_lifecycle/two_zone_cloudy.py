"""Dense-clump plus diffuse-region Cloudy emission model.

The mixing parameter is physical: ``dense_fraction`` is the fraction of the
incident hydrogen-ionizing photon rate intercepted by the dense phase.  Line
energies per incident photon are mixed before any diagnostic ratio is formed.
This avoids the unphysical practice of averaging ratios directly.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from run_cloudy_lrd_pilot import diagnostics


HERE = Path(__file__).resolve().parent
PILOT = HERE / "cloudy_lrd_pilot"
RUNS = PILOT / "runs"
DEFAULT_DENSE = "baseline"
DEFAULT_DIFFUSE_MODELS = (
    "diffuse_z_0p02",
    "diffuse_base",
    "diffuse_z_solar",
    "diffuse_u_low",
    "diffuse_u_high",
)


def load_model(name, runs=RUNS):
    """Load one verified Cloudy result by model name."""
    path = Path(runs) / name / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing Cloudy result: {path}")
    result = json.loads(path.read_text())
    if not result.get("status", {}).get("cloudy_exited_ok"):
        raise ValueError(f"Cloudy model {name!r} did not finish successfully")
    return result


def mix_yields(dense_yields, diffuse_yields, dense_fraction):
    """Mix line yields using the ionizing-photon interception fraction."""
    fraction = float(dense_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("dense_fraction must lie in [0, 1]")
    if set(dense_yields) != set(diffuse_yields):
        raise ValueError("dense and diffuse models must contain the same lines")
    return {
        line: fraction * float(dense_yields[line])
        + (1.0 - fraction) * float(diffuse_yields[line])
        for line in dense_yields
    }


def mixed_model(dense_name, diffuse_name, dense_fraction, runs=RUNS):
    """Return the yields, ratios, and provenance for one two-zone mixture."""
    dense = load_model(dense_name, runs=runs)
    diffuse = load_model(diffuse_name, runs=runs)
    yields = mix_yields(
        dense["erg_per_ionizing_photon"],
        diffuse["erg_per_ionizing_photon"],
        dense_fraction,
    )
    return {
        "dense_model": dense_name,
        "diffuse_model": diffuse_name,
        "dense_fraction": float(dense_fraction),
        "dense_parameters": dense["parameters"],
        "diffuse_parameters": diffuse["parameters"],
        "erg_per_ionizing_photon": yields,
        "diagnostics": diagnostics(yields),
    }


def sequence_rows(
    dense_name=DEFAULT_DENSE,
    diffuse_models=DEFAULT_DIFFUSE_MODELS,
    fractions=None,
    runs=RUNS,
):
    """Build a family of physically mixed line-yield sequences."""
    if fractions is None:
        fractions = np.linspace(0.0, 1.0, 101)
    rows = []
    for diffuse_name in diffuse_models:
        dense = load_model(dense_name, runs=runs)
        diffuse = load_model(diffuse_name, runs=runs)
        dense_params = dense["parameters"]
        diffuse_params = diffuse["parameters"]
        for fraction in fractions:
            yields = mix_yields(
                dense["erg_per_ionizing_photon"],
                diffuse["erg_per_ionizing_photon"],
                fraction,
            )
            row = {
                "dense_model": dense_name,
                "diffuse_model": diffuse_name,
                "dense_fraction": float(fraction),
                "dense_logn": dense_params["logn"],
                "dense_logu": dense_params["logu"],
                "dense_metallicity": dense_params["metallicity"],
                "diffuse_logn": diffuse_params["logn"],
                "diffuse_logu": diffuse_params["logu"],
                "diffuse_metallicity": diffuse_params["metallicity"],
                **diagnostics(yields),
            }
            row.update({f"yield_{line}": value for line, value in yields.items()})
            rows.append(row)
    return rows


def write_sequences(path=PILOT / "two_zone_sequences.csv",
                    dense_name=DEFAULT_DENSE,
                    diffuse_models=DEFAULT_DIFFUSE_MODELS,
                    fractions=None):
    """Write a requested dense/diffuse family and return its rows."""
    rows = sequence_rows(dense_name=dense_name,
                         diffuse_models=diffuse_models,
                         fractions=fractions)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    generated = write_sequences()
    print(f"wrote {len(generated)} mixtures to {PILOT / 'two_zone_sequences.csv'}")
