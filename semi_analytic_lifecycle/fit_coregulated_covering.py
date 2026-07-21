#!/usr/bin/env python3
"""Joint evaluation of the co-regulated covering law over a (gamma, k) grid.

For each lifecycle run of the covering scan this script computes:

- the physical-partition stack chi-square (full-competition and dust-quenched
  candidates from ``derive_interception_exponent.py``), i.e. how well the
  *un-imposed* photon partition reproduces the four stack-inverted dense
  fractions;
- the demographic chi-square from the run's own results JSON;
- the selected subtype fractions at 4.5 <= z <= 6.5;
- the emergent source-level exponent.

The joint objective is the plain sum of the stack and demographic
chi-squares.  This is a descriptive scan, not a posterior: the two terms have
different (and partly unpropagated) systematics, no covariance is included,
and the grid is coarse.  The scan identifies a preferred region and its
stability, nothing more.

Outputs: coregulated_covering_scan.json, coregulated_covering_scan.pdf/.png
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "interception", HERE / "derive_interception_exponent.py")
interception = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(interception)

GRID = [(g, k) for g in (1.5, 2.0, 2.5, 3.0) for k in (0.5, 1.0, 2.0)]
SUBTYPES = ["xLRD", "plusLRD", "minusLRD", "bLRD"]
CANDIDATE = "D_dust_quenched_912"   # headline physical partition
BASELINE_RESULTS = "lifecycle_results_physical_visibility_final.json"


def suffix_for(gamma: float, k: float) -> str:
    if (gamma, k) == (2.0, 1.0):
        return "coregulated_covering_mid"   # reused from the initial study
    return f"covscan_g{str(gamma).replace('.', 'p')}_k{str(k).replace('.', 'p')}"


def evaluate_catalog(catalog_name: str) -> dict | None:
    path = HERE / catalog_name
    if not path.exists():
        return None
    interception.CATALOG = catalog_name
    cat = interception.load()
    valid = (
        np.isfinite(cat["d"]) & (cat["d"] > 0)
        & np.isfinite(cat["t_escape"])
        & (cat["sfr"] > 0) & (cat["lbol"] > 0) & (cat["weight"] > 0)
    )
    for key in cat:
        cat[key] = cat[key][valid]
    weights = cat["p_select"] * cat["weight"]
    log_d = np.log10(cat["d"])
    ratios = interception.candidate_ratios(cat)
    window = (cat["z"] >= 4.5) & (cat["z"] <= 6.5)
    core = window & (np.abs(log_d) < 1.5)
    confront = interception.stack_confrontation(cat, ratios, weights)
    log_r = np.log10(np.clip(ratios[CANDIDATE], 1.0e-30, None))
    fit = interception.weighted_fit(log_d[core], log_r[core], weights[core])
    total = weights[window].sum()
    fractions = {
        s: float(weights[window & (cat["subtype"] == s)].sum() / total)
        for s in SUBTYPES
    }
    return {
        "stack_chi2": confront[CANDIDATE]["chi2_vs_stacks"],
        "stack_chi2_full_competition":
            confront["C_full_competition"]["chi2_vs_stacks"],
        "stack_medians": confront[CANDIDATE]["median_f_dense"],
        "eta_emergent": round(fit["eta_emergent"], 3),
        "subtype_fractions": {s: round(v, 4) for s, v in fractions.items()},
    }


def demographic_chi2(results_name: str) -> dict | None:
    path = HERE / results_name
    if not path.exists():
        return None
    fit = json.loads(path.read_text())["demographic_fit"]
    return {"chi2": fit["chi2"],
            "pulls": [round(p, 2) for p in fit["pulls_sigma"]]}


def main() -> None:
    rows = []
    for gamma, k in GRID:
        suffix = suffix_for(gamma, k)
        spectral = evaluate_catalog(f"synthetic_catalog_{suffix}.csv")
        demog = demographic_chi2(f"lifecycle_results_{suffix}.json")
        if spectral is None or demog is None:
            continue
        rows.append({
            "gamma": gamma, "k": k, "suffix": suffix,
            **spectral,
            "demographic_chi2": round(demog["chi2"], 2),
            "demographic_pulls": demog["pulls"],
            "joint_objective": round(spectral["stack_chi2"] + demog["chi2"], 2),
        })
    baseline = {
        "stack_chi2": 3.78,   # dust-quenched candidate, independent covering
        "demographic": demographic_chi2(BASELINE_RESULTS),
    }
    rows.sort(key=lambda r: r["joint_objective"])
    best = rows[0] if rows else None

    payload = {
        "candidate_partition": CANDIDATE,
        "objective": "stack_chi2 (physical partition) + demographic chi2; "
                     "descriptive sum, not a likelihood",
        "grid": rows,
        "baseline_independent_covering": baseline,
        "best": best,
    }
    (HERE / "coregulated_covering_scan.json").write_text(
        json.dumps(payload, indent=2) + "\n")

    if rows:
        gammas = sorted({r["gamma"] for r in rows})
        ks = sorted({r["k"] for r in rows})
        grid_vals = np.full((len(ks), len(gammas)), np.nan)
        for r in rows:
            grid_vals[ks.index(r["k"]), gammas.index(r["gamma"])] = \
                r["joint_objective"]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        im = axes[0].imshow(grid_vals, origin="lower", aspect="auto",
                            cmap="viridis_r")
        axes[0].set_xticks(range(len(gammas)), [str(g) for g in gammas])
        axes[0].set_yticks(range(len(ks)), [str(k) for k in ks])
        axes[0].set_xlabel(r"covering odds index $\gamma$")
        axes[0].set_ylabel(r"normalization $k$")
        axes[0].set_title("joint objective (stack + demographic $\\chi^2$)")
        for r in rows:
            axes[0].text(gammas.index(r["gamma"]), ks.index(r["k"]),
                         f"{r['joint_objective']:.1f}", ha="center",
                         va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=axes[0])

        axes[1].scatter([r["stack_chi2"] for r in rows],
                        [r["demographic_chi2"] for r in rows],
                        c=[r["gamma"] for r in rows], cmap="plasma", s=60)
        for r in rows:
            axes[1].annotate(f"$\\gamma$={r['gamma']},k={r['k']}",
                             (r["stack_chi2"], r["demographic_chi2"]),
                             fontsize=6, xytext=(3, 3),
                             textcoords="offset points")
        axes[1].axvline(baseline["stack_chi2"], color="0.5", ls="--", lw=1,
                        label="baseline stack $\\chi^2$")
        axes[1].axhline(baseline["demographic"]["chi2"], color="0.5", ls=":",
                        lw=1, label="baseline demographic $\\chi^2$")
        axes[1].set_xlabel("physical-partition stack $\\chi^2$")
        axes[1].set_ylabel("demographic $\\chi^2$")
        axes[1].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(HERE / "coregulated_covering_scan.pdf")
        fig.savefig(HERE / "coregulated_covering_scan.png", dpi=200)
        plt.close(fig)

    print(json.dumps({"best": best,
                      "n_evaluated": len(rows)}, indent=2))
    for r in rows:
        print(f"gamma={r['gamma']:>4} k={r['k']:>4}  stack={r['stack_chi2']:6.2f} "
              f"demog={r['demographic_chi2']:6.2f}  joint={r['joint_objective']:6.2f}  "
              f"eta={r['eta_emergent']:5.2f}  fr={list(r['subtype_fractions'].values())}")


if __name__ == "__main__":
    main()
