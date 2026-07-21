"""Confront the two-zone Cloudy sequences with the JADES DR4 parent sample.

This is deliberately a control-sample check, not an LRD parameter fit.  It
uses reported fluxes even below the catalogue detection threshold, avoiding
the upward bias from fitting detected line ratios only.  A one-sided term is
available for rows that provide an upper limit without a flux measurement.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from two_zone_cloudy import PILOT, sequence_rows, write_sequences


HERE = Path(__file__).resolve().parent
CATALOGUE = HERE.parent / "catalog_likelihood" / "jades_dr4_spectral_likelihood_table.csv"
SUMMARY = PILOT / "two_zone_control_fit.json"
FIGURE_PDF = PILOT / "two_zone_control_confrontation.pdf"
FIGURE_PNG = PILOT / "two_zone_control_confrontation.png"
SYSTEMATIC_FRACTION = 0.20
THERMAL_DIFFUSE_MODELS = (
    "thermal_skin_z_0p02",
    "thermal_skin_base",
    "thermal_skin_z_solar",
    "thermal_skin_u_mid",
    "thermal_skin_u_low",
    "thermal_skin_u_low_solar",
)
CHANNELS = {
    "oiii_hbeta": ("O3_5007", "HBaB_4861"),
    "nii_halpha": ("N2_6584", "HBaA_6563"),
    "halpha_hbeta": ("HBaA_6563", "HBaB_4861"),
}


def finite(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def selected_parent_rows(path=CATALOGUE, z_min=4.5, z_max=6.5, muv_max=-18.5):
    """Read the explicitly defined high-redshift luminous parent control."""
    selected = []
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            z = finite(row.get("z_spec"))
            muv = finite(row.get("muv"))
            if z is None or muv is None:
                continue
            if z_min <= z <= z_max and muv <= muv_max:
                selected.append(row)
    return selected


def build_measurements(rows, numerator, denominator):
    """Construct unthresholded-flux or censored measurements for one ratio."""
    measurements = []
    for row in rows:
        denominator_flux = finite(row.get(f"{denominator}_flux"))
        denominator_error = finite(row.get(f"{denominator}_error"))
        denominator_detected = row.get(f"{denominator}_detected") == "1"
        if (not denominator_detected or denominator_flux is None
                or denominator_error is None or denominator_flux <= 0
                or denominator_error <= 0):
            continue
        numerator_flux = finite(row.get(f"{numerator}_flux"))
        numerator_error = finite(row.get(f"{numerator}_error"))
        upper = finite(row.get(f"{numerator}_upper3sigma"))
        common = {
            "unique_id": row["unique_id"],
            "denominator_flux": denominator_flux,
            "denominator_error": denominator_error,
            "numerator_detected": row.get(f"{numerator}_detected") == "1",
        }
        if numerator_flux is not None and numerator_error is not None and numerator_error > 0:
            measurements.append({
                **common,
                "mode": "flux",
                "numerator_flux": numerator_flux,
                "numerator_error": numerator_error,
            })
        elif upper is not None and upper > 0:
            measurements.append({**common, "mode": "upper", "upper": upper})
    return measurements


def gaussian_cdf(value):
    """Numerically safe standard-normal CDF for scalar values."""
    return max(0.5 * math.erfc(-float(value) / math.sqrt(2.0)), 1.0e-300)


def channel_deviance(predicted_ratio, measurements,
                     systematic_fraction=SYSTEMATIC_FRACTION):
    """Mean -2 log-likelihood contribution, excluding irrelevant constants."""
    if predicted_ratio < 0:
        raise ValueError("predicted line ratio cannot be negative")
    terms = []
    for measurement in measurements:
        mean = predicted_ratio * measurement["denominator_flux"]
        denominator_variance = (
            predicted_ratio * measurement["denominator_error"]
        ) ** 2
        model_variance = (systematic_fraction * mean) ** 2
        if measurement["mode"] == "flux":
            variance = (measurement["numerator_error"] ** 2
                        + denominator_variance + model_variance)
            terms.append((measurement["numerator_flux"] - mean) ** 2 / variance)
        else:
            limit_sigma = measurement["upper"] / 3.0
            sigma = math.sqrt(limit_sigma**2 + denominator_variance + model_variance)
            terms.append(-2.0 * math.log(
                gaussian_cdf((measurement["upper"] - mean) / sigma)
            ))
    return float(np.mean(terms)) if terms else math.nan


def score_candidate(candidate, measurements):
    """Give each diagnostic equal weight so sample size cannot dominate."""
    channel_scores = {
        channel: channel_deviance(candidate[channel], data)
        for channel, data in measurements.items()
    }
    finite_scores = [value for value in channel_scores.values() if math.isfinite(value)]
    return float(sum(finite_scores)), channel_scores


def detected_ratio_quantiles(rows, numerator, denominator):
    ratios = []
    for row in rows:
        if (row.get(f"{numerator}_detected") != "1"
                or row.get(f"{denominator}_detected") != "1"):
            continue
        top = finite(row.get(f"{numerator}_flux"))
        bottom = finite(row.get(f"{denominator}_flux"))
        if top is not None and bottom is not None and top > 0 and bottom > 0:
            ratios.append(top / bottom)
    if not ratios:
        return {"count": 0, "p16": None, "median": None, "p84": None}
    p16, median, p84 = np.quantile(ratios, [0.16, 0.50, 0.84])
    return {"count": len(ratios), "p16": float(p16),
            "median": float(median), "p84": float(p84)}


def measurement_counts(measurements):
    return {
        channel: {
            "total": len(data),
            "flux_likelihood": sum(item["mode"] == "flux" for item in data),
            "censored_upper_limit": sum(item["mode"] == "upper" for item in data),
            "numerator_detected": sum(item["numerator_detected"] for item in data),
            "numerator_nondetected": sum(not item["numerator_detected"] for item in data),
        }
        for channel, data in measurements.items()
    }


def compact_result(candidate, objective, channel_scores):
    return {
        "diffuse_model": candidate["diffuse_model"],
        "dense_fraction_qh": candidate["dense_fraction"],
        "diffuse_logu": candidate["diffuse_logu"],
        "diffuse_metallicity_zsun": candidate["diffuse_metallicity"],
        "objective_equal_channel_mean_deviance": objective,
        "channel_mean_deviance": channel_scores,
        "predicted_ratios": {
            name: candidate[name] for name in CHANNELS
        } | {
            "sii_halpha": candidate["sii_halpha"],
            "heii1640_hbeta": candidate["heii1640_hbeta"],
        },
        "lha_per_qh54": candidate["lha_per_qh54"],
    }


def fit_control(rows=None, candidates=None):
    if rows is None:
        rows = selected_parent_rows()
    if candidates is None:
        candidates = sequence_rows()
    measurements = {
        channel: build_measurements(rows, numerator, denominator)
        for channel, (numerator, denominator) in CHANNELS.items()
    }
    scored = []
    for candidate in candidates:
        objective, channel_scores = score_candidate(candidate, measurements)
        scored.append((objective, candidate, channel_scores))
    scored.sort(key=lambda item: item[0])

    profiles = []
    grouped = defaultdict(list)
    for result in scored:
        grouped[result[1]["diffuse_model"]].append(result)
    for diffuse_model, results in grouped.items():
        objective, candidate, channel_scores = min(results, key=lambda item: item[0])
        profiles.append(compact_result(candidate, objective, channel_scores))
    profiles.sort(key=lambda item: item["objective_equal_channel_mean_deviance"])

    best_objective, best_candidate, best_scores = scored[0]
    return {
        "scope": (
            "Descriptive confrontation with a UV-luminous JADES DR4 spectroscopic "
            "parent/control sample. The table has no LRD-specific classification, "
            "so this is not an LRD posterior or a calibrated population inference."
        ),
        "selection": {
            "z_spec_min": 4.5,
            "z_spec_max": 6.5,
            "muv_max_brightward": -18.5,
            "selected_parent_objects": len(rows),
        },
        "likelihood": {
            "method": (
                "Gaussian likelihood on reported numerator flux conditional on a "
                "detected denominator; nondetected numerator fluxes are retained. "
                "A one-sided Gaussian CDF is used only if flux is absent but a "
                "3-sigma upper limit exists."
            ),
            "fractional_model_and_cross_band_floor": SYSTEMATIC_FRACTION,
            "objective": (
                "sum over diagnostics of their mean -2 log-likelihood term; "
                "diagnostics receive equal weight"
            ),
            "counts": measurement_counts(measurements),
        },
        "parameter_definition": (
            "dense_fraction_qh is the fraction of incident Q(H) intercepted by "
            "dense clumps; it is not a gas-mass or volume fraction."
        ),
        "best_control_match": compact_result(best_candidate, best_objective, best_scores),
        "best_by_diffuse_hypothesis": profiles,
        "detected_ratio_quantiles_for_visual_reference_only": {
            channel: detected_ratio_quantiles(rows, numerator, denominator)
            for channel, (numerator, denominator) in CHANNELS.items()
        },
        "interpretation_limit": (
            "Detected-ratio quantiles are visualization aids and are not used for "
            "fitting because thresholding biases them high. Model rankings only "
            "test whether the proposed sequence overlaps an ordinary high-z "
            "spectral locus; an LRD-labelled spectral sample is still required."
        ),
    }, scored


def _joint_detected(rows, pairs):
    points = []
    for row in rows:
        values = []
        valid = True
        for numerator, denominator in pairs:
            if (row.get(f"{numerator}_detected") != "1"
                    or row.get(f"{denominator}_detected") != "1"):
                valid = False
                break
            top = finite(row.get(f"{numerator}_flux"))
            bottom = finite(row.get(f"{denominator}_flux"))
            if top is None or bottom is None or top <= 0 or bottom <= 0:
                valid = False
                break
            values.append(top / bottom)
        if valid:
            points.append(values)
    return np.asarray(points, dtype=float)


def make_figure(rows, candidates, best, alternate=None,
                tier_label="thermal-balance skin"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diffuse_names = list(dict.fromkeys(row["diffuse_model"] for row in candidates))
    labels = {}
    for name in diffuse_names:
        example = next(row for row in candidates if row["diffuse_model"] == name)
        labels[name] = (f"Z={example['diffuse_metallicity']:g}, "
                        f"log U={example['diffuse_logu']:g}")
    colors = dict(zip(diffuse_names,
                      plt.cm.viridis(np.linspace(0.05, 0.9, len(diffuse_names)))))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

    bpt = _joint_detected(rows, (("N2_6584", "HBaA_6563"),
                                 ("O3_5007", "HBaB_4861")))
    if bpt.size:
        axes[0].scatter(np.log10(bpt[:, 0]), np.log10(bpt[:, 1]), s=13,
                        color="0.72", alpha=0.50, edgecolors="none",
                        label=f"JADES parent, joint detections (N={len(bpt)})")
    balmer = _joint_detected(rows, (("HBaA_6563", "HBaB_4861"),
                                    ("O3_5007", "HBaB_4861")))
    if balmer.size:
        axes[1].scatter(balmer[:, 0], np.log10(balmer[:, 1]), s=13,
                        color="0.72", alpha=0.50, edgecolors="none",
                        label=f"JADES parent, joint detections (N={len(balmer)})")

    for diffuse, color in colors.items():
        sequence = sorted(
            (row for row in candidates if row["diffuse_model"] == diffuse),
            key=lambda row: row["dense_fraction"],
        )
        x_bpt = np.log10([row["nii_halpha"] for row in sequence])
        y = np.log10([row["oiii_hbeta"] for row in sequence])
        axes[0].plot(x_bpt, y, color=color, lw=2, label=labels[diffuse])
        axes[1].plot([row["halpha_hbeta"] for row in sequence], y,
                     color=color, lw=2, label=labels[diffuse])
        # Mark pure diffuse, half, and nearly pure dense positions.
        for index in (0, 50, 99):
            axes[0].scatter(x_bpt[index], y[index], color=color, s=16, zorder=3)
            axes[1].scatter(sequence[index]["halpha_hbeta"], y[index],
                            color=color, s=16, zorder=3)

    best_ratios = best["predicted_ratios"]
    axes[0].scatter(math.log10(best_ratios["nii_halpha"]),
                    math.log10(best_ratios["oiii_hbeta"]), marker="*", s=150,
                    color="#d62728", edgecolor="white", linewidth=0.8,
                    label="numerical minimum (degenerate)", zorder=5)
    axes[1].scatter(best_ratios["halpha_hbeta"],
                    math.log10(best_ratios["oiii_hbeta"]), marker="*", s=150,
                    color="#d62728", edgecolor="white", linewidth=0.8, zorder=5)
    if alternate is not None:
        alternate_ratios = alternate["predicted_ratios"]
        axes[0].scatter(math.log10(alternate_ratios["nii_halpha"]),
                        math.log10(alternate_ratios["oiii_hbeta"]), marker="D", s=70,
                        color="#1f77b4", edgecolor="white", linewidth=0.8,
                        label="parsimonious tied solution", zorder=5)
        axes[1].scatter(alternate_ratios["halpha_hbeta"],
                        math.log10(alternate_ratios["oiii_hbeta"]), marker="D", s=70,
                        color="#1f77b4", edgecolor="white", linewidth=0.8, zorder=5)

    axes[0].set(xlabel=r"log([N II] 6584 / H$\alpha$)",
                ylabel=r"log([O III] 5007 / H$\beta$)",
                title="Metal-line locus")
    axes[1].set(xlabel=r"H$\alpha$ / H$\beta$",
                ylabel=r"log([O III] 5007 / H$\beta$)",
                title="Balmer and metal-line consistency")
    axes[1].set_xlim(1.0, 6.0)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7.4, loc="best")
    figure.suptitle(
        f"Two-zone Cloudy {tier_label} sequences vs. a high-z JADES parent control\n"
        "points on curves mark dense Q(H) fractions 0, 0.5, and 0.99",
        fontsize=11,
    )
    figure.savefig(FIGURE_PDF)
    figure.savefig(FIGURE_PNG, dpi=180)
    plt.close(figure)


def main():
    fixed_candidates = write_sequences()
    thermal_candidates = write_sequences(
        path=PILOT / "two_zone_thermal_sequences.csv",
        dense_name="thermal_dense_skin",
        diffuse_models=THERMAL_DIFFUSE_MODELS,
    )
    rows = selected_parent_rows()
    fixed_summary, _ = fit_control(rows=rows, candidates=fixed_candidates)
    thermal_summary, _ = fit_control(rows=rows, candidates=thermal_candidates)
    fixed_best = fixed_summary["best_control_match"]
    thermal_best = thermal_summary["best_control_match"]
    thermal_profiles = thermal_summary["best_by_diffuse_hypothesis"]
    parsimonious = next(
        row for row in thermal_profiles
        if row["diffuse_model"] == "thermal_skin_u_mid"
    )
    profile_delta = (
        parsimonious["objective_equal_channel_mean_deviance"]
        - thermal_best["objective_equal_channel_mean_deviance"]
    )
    summary = {
        "preferred_tier": "thermal_balance_emitting_skin",
        "fixed_temperature_screening": fixed_summary,
        "thermal_balance_emitting_skin": thermal_summary,
        "comparison": {
            "fixed_temperature_best_objective":
                fixed_best["objective_equal_channel_mean_deviance"],
            "thermal_balance_best_objective":
                thermal_best["objective_equal_channel_mean_deviance"],
            "objective_improvement":
                fixed_best["objective_equal_channel_mean_deviance"]
                - thermal_best["objective_equal_channel_mean_deviance"],
            "parsimonious_subsolar_diffuse_reference": parsimonious,
            "parsimonious_delta_objective_from_numerical_best": profile_delta,
            "dense_fraction_identified_by_parent_control": False,
            "conclusion": (
                "Thermal balance supplies a subsolar, purely diffuse solution "
                "that is effectively tied with the numerical solar/high-dense "
                "minimum. The fixed-temperature solar preference was therefore "
                "not robust, and the parent control does not identify metallicity "
                "or dense fraction. Only an LRD-labelled sample with decomposed "
                "broad and narrow lines can do so."
            ),
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    make_figure(rows, thermal_candidates, thermal_best, alternate=parsimonious)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
