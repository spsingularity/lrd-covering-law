#!/usr/bin/env python3
"""Letter Table 2 and the z~9.5 headroom, from the fitted-law lifecycle run.

Reads ``lifecycle_results_coregulated_covering_mid.json`` (which carries the
selection-free intrinsic densities of the nuclear-active M_UV < -18.5
population, emitted inside the simulation because the catalogue's
p_select > 1e-6 storage floor cannot recover them), applies the fitted
abundance normalization from the demographic fit, and emits:

- per-subtype intrinsic and selected densities at z ~ 9.5 (Letter Table 2);
- the gate-passing fraction per subtype;
- the total intrinsic headroom over the reported broad-selection density
  (the "~34x" of main-paper Sec. 6 / Letter Sec. 5);
- the selected-density deficit factor.

Writes ``highz_gate_fractions.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "lifecycle_results_coregulated_covering_mid.json"
REPORTED_Z95 = 3.18e-5      # Rinaldi et al. broad-selection density at z~9.5
SUBTYPES = ("xLRD", "plusLRD", "minusLRD", "bLRD")


def main():
    data = json.loads(RESULTS.read_text())
    fit = data["demographic_fit"]
    summaries = data["summaries"]

    row = min(summaries, key=lambda r: abs(r["z"] - 9.5))
    if "density_active_muv185_intrinsic" not in row:
        raise SystemExit(
            "results json predates the intrinsic-density emit; regenerate with "
            "lrd_lifecycle.py --physical-visibility --coregulated-covering "
            "--covering-odds-index 2.0 --covering-odds-normalization 1.0 "
            "--dt-gyr 0.006 --early-burst-rate 12.0 "
            "--suffix coregulated_covering_mid")

    # The demographic fit applies one abundance normalization to raw model
    # densities; recover it from any bin (predicted / raw at that z).
    zi = int(np.argmin(np.abs(np.array(fit["observed_z"]) - 9.5)))
    raw95 = row["raw_density"]
    norm = fit["predicted_density"][zi] / raw95

    out = {
        "scope": ("fitted-law model (coregulated 2,1 + physical visibility), "
                  "z~9.5 snapshot, M_UV < -18.5; densities carry the fitted "
                  "abundance normalization"),
        "abundance_normalization": norm,
        "reported_broad_selection_density": REPORTED_Z95,
        "selected_density": fit["predicted_density"][zi],
        "selected_deficit_factor": REPORTED_Z95 / fit["predicted_density"][zi],
        "intrinsic_muv185_density": row["density_active_muv185_intrinsic"] * norm,
        "intrinsic_headroom_factor":
            row["density_active_muv185_intrinsic"] * norm / REPORTED_Z95,
        "per_subtype": {},
    }
    for st in SUBTYPES:
        intr = row.get(f"density_muv185_intrinsic_{st}", 0.0) * norm
        sel = row.get(f"density_muv185_selected_{st}", 0.0) * norm
        out["per_subtype"][st] = {
            "intrinsic_density": intr,
            "selected_density": sel,
            "pass_fraction": (sel / intr) if intr > 0 else None,
        }
    (HERE / "highz_gate_fractions.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
