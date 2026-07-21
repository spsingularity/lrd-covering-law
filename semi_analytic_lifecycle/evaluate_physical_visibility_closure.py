#!/usr/bin/env python3
"""Evaluate the source-level physical visibility closure against LRD stacks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit


HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "public_lrd_constraints"
SUBTYPES = ["xLRD", "plusLRD", "minusLRD", "bLRD"]
OBSERVED_SUBTYPE_FRACTIONS = {
    "xLRD": 0.12, "plusLRD": 0.31, "minusLRD": 0.31, "bLRD": 0.27,
}


def weighted_quantile(values, weights, q=0.5):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[good], weights[good]
    if not len(values):
        return np.nan
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    centers = np.cumsum(weights) - 0.5 * weights
    return float(np.interp(q * weights.sum(), centers, values))


def dense_fraction_from_dominance(dominance, coupling_index):
    log_odds = coupling_index * np.log(np.clip(dominance, 1e-12, 1e12))
    return expit(log_odds)


def subtype_medians(frame, column):
    return {
        subtype: weighted_quantile(
            frame.loc[frame.visibility_subtype == subtype, column],
            frame.loc[frame.visibility_subtype == subtype, "weight_cMpc3"],
        )
        for subtype in SUBTYPES
    }


def load_dense_constraints():
    fits = pd.read_csv(PUBLIC / "public_lrd_stack_two_zone_fits.csv")
    fits = fits[fits.temperature_basis_k == 50000].copy()
    result = {}
    for row in fits.itertuples():
        interval = json.loads(row.dense_qh_fraction_measurement_p16_p50_p84)
        result[row.subtype] = {
            "central": float(row.dense_qh_fraction),
            "p16": float(interval[0]),
            "p84": float(interval[2]),
        }
    return result


def asymmetric_chi2(prediction, constraints, floor=0.0, omit=None):
    total = 0.0
    for subtype in SUBTYPES:
        if subtype == omit:
            continue
        pred = prediction[subtype]
        obs = constraints[subtype]["central"]
        sigma = (constraints[subtype]["p84"] - obs if pred >= obs
                 else obs - constraints[subtype]["p16"])
        sigma = max(sigma, floor, 1e-8)
        total += ((pred - obs) / sigma) ** 2
    return float(total)


def eta_prediction(frame, eta):
    fractions = dense_fraction_from_dominance(
        frame.visibility_core_to_host_5100.to_numpy(), eta
    )
    working = frame.assign(_dense_eta=fractions)
    return subtype_medians(working, "_dense_eta")


def scan_eta(frame, constraints, floor=0.0, omit=None):
    grid = np.linspace(0.5, 4.0, 701)
    chi2 = np.array([
        asymmetric_chi2(eta_prediction(frame, eta), constraints,
                        floor=floor, omit=omit)
        for eta in grid
    ])
    index = int(np.argmin(chi2))
    return grid, chi2, float(grid[index]), eta_prediction(frame, grid[index])


def subtype_fractions(frame, weights):
    weights = np.asarray(weights, float)
    total = weights.sum()
    return {
        subtype: float(weights[frame.visibility_subtype.to_numpy() == subtype].sum()
                       / total)
        for subtype in SUBTYPES
    }


def selection_stress_test(frame):
    # Recover the common selection factors, then replace only the two
    # single-band flux gates. This brackets sensitivity; it is not a fitted
    # survey completeness function.
    uv_flux_gate = (
        expit((-18.5 - frame.muv.to_numpy()) / 0.25)
        * expit((27.5 - (frame.muv.to_numpy() + 47.0
                         - 5.0 * np.log10(1.0 + frame.z.to_numpy()))) / 0.7)
    )
    optical_flux_gate = (
        expit((-18.5 - frame.visibility_m5100.to_numpy()) / 0.25)
        * expit((27.5 - (frame.visibility_m5100.to_numpy() + 47.0
                         - 5.0 * np.log10(1.0 + frame.z.to_numpy()))) / 0.7)
    )
    selected_weight = frame.weight_cMpc3.to_numpy()
    probability = np.clip(frame.p_select.to_numpy(), 1e-30, None)
    raw_weight = selected_weight / probability
    common_gate = probability / np.clip(uv_flux_gate, 1e-30, None)
    optical_weight = raw_weight * common_gate * optical_flux_gate
    support_weight = raw_weight * common_gate
    return {
        "current_uv_gated": subtype_fractions(frame, selected_weight),
        "optical_gated_stress_test": subtype_fractions(frame, optical_weight),
        "pre_flux_gate_support": subtype_fractions(frame, support_weight),
        "published_stack_sample": OBSERVED_SUBTYPE_FRACTIONS,
    }


def seed_stability(catalog_paths, z_lo, z_hi):
    rows = []
    for path in catalog_paths:
        trial = pd.read_csv(path)
        trial = trial[(trial.z >= z_lo) & (trial.z <= z_hi)].copy()
        medians = subtype_medians(trial, "visibility_dense_fraction_qh")
        fractions = subtype_fractions(trial, trial.weight_cMpc3.to_numpy())
        row = {"catalog": Path(path).name}
        for subtype in SUBTYPES:
            row[f"dense_fraction_{subtype}"] = medians[subtype]
            row[f"population_fraction_{subtype}"] = fractions[subtype]
        rows.append(row)
    table = pd.DataFrame(rows)
    metric_columns = [column for column in table if column != "catalog"]
    ranges = {
        column: {
            "minimum": float(table[column].min()),
            "median": float(table[column].median()),
            "maximum": float(table[column].max()),
        }
        for column in metric_columns
    }
    return table, ranges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        default=str(HERE / "synthetic_catalog_physical_visibility_final.csv"),
    )
    parser.add_argument("--z-lo", type=float, default=4.5)
    parser.add_argument("--z-hi", type=float, default=6.5)
    args = parser.parse_args()

    frame = pd.read_csv(args.catalog)
    frame = frame[(frame.z >= args.z_lo) & (frame.z <= args.z_hi)].copy()
    if not len(frame):
        raise ValueError("catalog has no rows in the requested redshift interval")
    constraints = load_dense_constraints()

    reference_prediction = subtype_medians(
        frame, "visibility_dense_fraction_qh"
    )
    grid, chi2_measurement, eta_measurement, prediction_measurement = scan_eta(
        frame, constraints, floor=0.0
    )
    _, chi2_systematic, eta_systematic, prediction_systematic = scan_eta(
        frame, constraints, floor=0.05
    )
    leave_one_out = {}
    for subtype in SUBTYPES:
        _, _, eta, prediction = scan_eta(
            frame, constraints, floor=0.05, omit=subtype
        )
        leave_one_out[subtype] = {
            "best_coupling_index": eta,
            "held_out_prediction": prediction[subtype],
            "held_out_observed": constraints[subtype]["central"],
            "held_out_p16": constraints[subtype]["p16"],
            "held_out_p84": constraints[subtype]["p84"],
        }

    marks = {
        key: subtype_medians(frame, key)
        for key in [
            "visibility_l5100_l2500", "visibility_core_to_host_5100",
            "visibility_dense_fraction_qh", "visibility_qh_escape_fraction",
            "visibility_oiii_hbeta", "visibility_oiii_oii",
            "visibility_heii4686_hbeta", "visibility_halpha_hbeta",
            "visibility_hbeta_hgamma", "log_lx", "variability_rms",
        ]
    }
    observed_lines = pd.read_csv(
        PUBLIC / "perez_gonzalez26_lrd_stack_constraints.csv"
    ).set_index("subtype")
    line_chi2 = 0.0
    for subtype in SUBTYPES:
        for model_key, obs_key, err_key in [
            ("visibility_oiii_hbeta", "oiii5007_hbeta",
             "oiii5007_hbeta_error"),
            ("visibility_oiii_oii", "oiii5007_oii3727",
             "oiii5007_oii3727_error"),
        ]:
            line_chi2 += (
                (marks[model_key][subtype] - observed_lines.loc[subtype, obs_key])
                / observed_lines.loc[subtype, err_key]
            ) ** 2

    dense_rows = []
    for subtype in SUBTYPES:
        dense_rows.append({
            "subtype": subtype,
            "observed_dense_fraction": constraints[subtype]["central"],
            "observed_p16": constraints[subtype]["p16"],
            "observed_p84": constraints[subtype]["p84"],
            "predicted_dense_fraction_eta2": reference_prediction[subtype],
            "predicted_oiii_hbeta": marks["visibility_oiii_hbeta"][subtype],
            "observed_oiii_hbeta": observed_lines.loc[subtype, "oiii5007_hbeta"],
            "predicted_oiii_oii": marks["visibility_oiii_oii"][subtype],
            "observed_oiii_oii": observed_lines.loc[subtype, "oiii5007_oii3727"],
            "predicted_heii4686_hbeta": marks["visibility_heii4686_hbeta"][subtype],
        })
    dense_table = pd.DataFrame(dense_rows)
    dense_table.to_csv(HERE / "physical_visibility_stack_predictions.csv", index=False)

    selection = selection_stress_test(frame)
    seed_paths = [Path(args.catalog)] + sorted(
        HERE.glob("synthetic_catalog_physical_visibility_seed*.csv")
    )
    stability_table, stability_ranges = seed_stability(
        seed_paths, args.z_lo, args.z_hi
    )
    stability_table.to_csv(
        HERE / "physical_visibility_seed_stability.csv", index=False
    )
    summary = {
        "catalog": str(Path(args.catalog).resolve()),
        "redshift_interval": [args.z_lo, args.z_hi],
        "sources_in_catalog_support": int(len(frame)),
        "reference_quadratic_closure": {
            "coupling_index": 2.0,
            "predicted_dense_fraction_by_subtype": reference_prediction,
            "measurement_only_chi2": asymmetric_chi2(
                reference_prediction, constraints, floor=0.0
            ),
            "chi2_with_0p05_model_floor": asymmetric_chi2(
                reference_prediction, constraints, floor=0.05
            ),
        },
        "coupling_index_scan": {
            "measurement_only_best": eta_measurement,
            "measurement_only_prediction": prediction_measurement,
            "with_0p05_model_floor_best": eta_systematic,
            "with_0p05_model_floor_prediction": prediction_systematic,
            "leave_one_subtype_out_with_0p05_floor": leave_one_out,
        },
        "fixed_soft_sed_line_test": {
            "oiii_two_ratio_chi2": float(line_chi2),
            "observables": 8,
            "fitted_line_parameters_in_reference_prediction": 0,
            "all_predicted_heii4686_hbeta_below_0p1": bool(
                all(value < 0.1 for value
                    in marks["visibility_heii4686_hbeta"].values())
            ),
        },
        "predicted_marks_by_subtype": marks,
        "selection_stress_test": selection,
        "seed_stability": {
            "catalogs": [str(path.resolve()) for path in seed_paths],
            "ranges": stability_ranges,
        },
        "interpretive_limits": [
            "The stack dense fractions are themselves two-line inversions, not direct measurements.",
            "The 0.05 floor is a declared model-systematics allowance, not a fitted uncertainty.",
            "The optical-gated case brackets selection sensitivity and is not an injection-recovery calibration.",
            "Balmer ratios listed here are intrinsic; comparison to observed ratios requires a transfer likelihood.",
        ],
    }
    (HERE / "physical_visibility_closure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    x = np.arange(len(SUBTYPES))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    ax = axes[0, 0]
    obs = np.array([constraints[s]["central"] for s in SUBTYPES])
    low = obs - np.array([constraints[s]["p16"] for s in SUBTYPES])
    high = np.array([constraints[s]["p84"] for s in SUBTYPES]) - obs
    pred = np.array([reference_prediction[s] for s in SUBTYPES])
    ax.errorbar(x, obs, yerr=[low, high], fmt="o", color="black",
                capsize=3, label="stack inversion")
    ax.plot(x, pred, "s-", color="#b54a4a", label=r"quadratic closure")
    ax.set(ylabel=r"dense fraction of processed $Q(H)$", ylim=(-0.04, 1.05))
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for key, color, label in [
        ("visibility_oiii_hbeta", "#2864a5", r"[O III]/H$\beta$"),
        ("visibility_oiii_oii", "#d17a22", r"[O III]/[O II]"),
    ]:
        ax.plot(x, [marks[key][s] for s in SUBTYPES], "s-",
                color=color, label=label + " model")
    ax.errorbar(x, observed_lines.loc[SUBTYPES, "oiii5007_hbeta"],
                yerr=observed_lines.loc[SUBTYPES, "oiii5007_hbeta_error"],
                fmt="o", color="#2864a5", mfc="white")
    ax.errorbar(x, observed_lines.loc[SUBTYPES, "oiii5007_oii3727"],
                yerr=observed_lines.loc[SUBTYPES, "oiii5007_oii3727_error"],
                fmt="o", color="#d17a22", mfc="white")
    ax.set(ylabel="line ratio", ylim=(0, 35))
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    width = 0.25
    for offset, key, color, label in [
        (-width, "published_stack_sample", "black", "published stacks"),
        (0.0, "current_uv_gated", "#2864a5", "current UV gate"),
        (width, "optical_gated_stress_test", "#d17a22", "optical gate stress test"),
    ]:
        ax.bar(x + offset, [selection[key][s] for s in SUBTYPES], width,
               color=color, alpha=0.8, label=label)
    ax.set(ylabel="subtype fraction", ylim=(0, 0.65))
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    ax.plot(grid, chi2_measurement, color="0.25", label="measurement only")
    ax.plot(grid, chi2_systematic, color="#3a913f", label="0.05 model floor")
    ax.axvline(2.0, color="#b54a4a", ls="--", label="quadratic closure")
    ax.set(xlabel="global coupling index", ylabel=r"$\chi^2$",
           xlim=(0.5, 4.0), ylim=(0, min(30, np.nanmax(chi2_systematic))))
    ax.legend(frameon=False, fontsize=7)

    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xticks(x, SUBTYPES, rotation=20)
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 1].set_xticks(np.arange(0.5, 4.1, 0.5))
    fig.suptitle("Physical visibility closure: predictions and stress tests")
    fig.tight_layout()
    fig.savefig(HERE / "physical_visibility_closure.png", dpi=220)
    plt.close(fig)

    print(json.dumps({
        "eta_measurement_only": eta_measurement,
        "eta_with_model_floor": eta_systematic,
        "quadratic_dense_chi2_with_floor": summary[
            "reference_quadratic_closure"
        ]["chi2_with_0p05_model_floor"],
        "fixed_soft_sed_oiii_chi2": float(line_chi2),
        "heii_gate_pass": summary["fixed_soft_sed_line_test"][
            "all_predicted_heii4686_hbeta_below_0p1"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
