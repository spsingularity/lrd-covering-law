"""Confront lifecycle Balmer widths with the public RUBIES LRD table.

This is a descriptive calibration of the compact line-emitting radius.  The
published RUBIES fitting prior caps the broad Gaussian FWHM below 2500 km/s,
so the result must not be interpreted as an unbiased BLR-radius posterior.
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np

from lrd_lifecycle import AU, G, MSUN, PC, Config, simulate, weighted_quantile


HERE = Path(__file__).resolve().parent
DATA = HERE / "public_lrd_constraints" / "rubies_hviding25_lrd_catalog.csv"
OUT = HERE / "public_lrd_constraints"
# Radius scan support [AU]; module-level so controlled reruns can widen it.
RADIUS_GRID_AU = (250.0, 5000.0, 401)


def read_rubies_redshift_slice(z_lo=4.5, z_hi=6.5):
    with DATA.open() as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if z_lo <= float(row["z"]) <= z_hi]


def redshift_matched_model_catalog(catalog, observed_redshifts):
    """Give each observed redshift equal weight in a nearest-snapshot mixture."""
    model_z = np.array([float(row["z"]) for row in catalog])
    unique_z = np.unique(model_z)
    selected_rows = []
    selected_weights = []
    for observed_z in observed_redshifts:
        snapshot_z = unique_z[np.argmin(np.abs(unique_z - observed_z))]
        indices = np.flatnonzero(model_z == snapshot_z)
        weights = np.array([float(catalog[index]["weight_cMpc3"])
                            for index in indices])
        if weights.sum() <= 0:
            continue
        selected_rows.extend(catalog[index] for index in indices)
        selected_weights.extend(weights / weights.sum() / len(observed_redshifts))
    return selected_rows, np.asarray(selected_weights)


def widths_at_radius(rows, radius_au):
    mass = np.array([10.0 ** float(row["logmbh"]) for row in rows])
    tau_e = np.array([float(row["tau_e"]) for row in rows])
    electron = 1100.0 * np.sqrt(np.clip(tau_e, 0.0, 12.0))
    virial = np.sqrt(G * mass * MSUN / (radius_au * AU)) / 1.0e5
    return np.sqrt(virial**2 + electron**2)


def legacy_reservoir_widths(rows):
    mass = np.array([10.0 ** float(row["logmbh"]) for row in rows])
    tau_e = np.array([float(row["tau_e"]) for row in rows])
    radius = np.maximum(np.array([float(row["reff_pc"]) for row in rows]), 0.3) * PC
    electron = 1100.0 * np.sqrt(np.clip(tau_e, 0.0, 12.0))
    return np.sqrt((np.sqrt(G * mass * MSUN / radius) / 1.0e5)**2 + electron**2)


def weighted_cdf(values, weights, grid):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return np.interp(grid, values, cumulative, left=0.0, right=1.0)


def empirical_cdf(values, grid):
    values = np.sort(np.asarray(values))
    return np.searchsorted(values, grid, side="right") / len(values)


def main():
    observed_rows = read_rubies_redshift_slice()
    observed_z = np.array([float(row["z"]) for row in observed_rows])
    observed = np.array([float(row["fwhm"]) for row in observed_rows])

    cfg = Config(
        n_halo=12000,
        seed=314159,
        dt_gyr=0.012,
        early_burst_rate_gyr=12.0,
        porous_envelope_enabled=True,
        clump_transfer_enabled=True,
    )
    result = simulate(cfg, store_catalog=True)
    model_rows, weights = redshift_matched_model_catalog(result["catalog"], observed_z)

    radii = np.geomspace(*RADIUS_GRID_AU)
    model_medians = np.array([
        weighted_quantile(widths_at_radius(model_rows, radius), weights)
        for radius in radii
    ])
    observed_median = float(np.median(observed))
    fitted_radius = float(radii[np.argmin(np.abs(model_medians - observed_median))])

    rng = np.random.default_rng(20260717)
    bootstrap_radii = []
    for _ in range(2000):
        sample_median = np.median(rng.choice(observed, len(observed), replace=True))
        bootstrap_radii.append(radii[np.argmin(np.abs(model_medians - sample_median))])
    radius_interval = np.quantile(bootstrap_radii, [0.16, 0.84])

    compact = widths_at_radius(model_rows, fitted_radius)
    legacy = legacy_reservoir_widths(model_rows)
    quantiles = [0.16, 0.5, 0.84]
    observed_quantiles = np.quantile(observed, quantiles)
    compact_quantiles = [weighted_quantile(compact, weights, q) for q in quantiles]
    legacy_quantiles = [weighted_quantile(legacy, weights, q) for q in quantiles]

    grid = np.linspace(0, 4000, 1001)
    obs_cdf = empirical_cdf(observed, grid)
    compact_cdf = weighted_cdf(compact, weights, grid)
    legacy_cdf = weighted_cdf(legacy, weights, grid)
    compact_ks = float(np.max(np.abs(obs_cdf - compact_cdf)))
    legacy_ks = float(np.max(np.abs(obs_cdf - legacy_cdf)))

    summary = {
        "scope": "RUBIES LRDs at 4.5 <= z <= 6.5; redshift-matched lifecycle snapshots",
        "observed_object_count": len(observed),
        "observed_fwhm_kms_p16_median_p84": observed_quantiles.tolist(),
        "published_fit_caution": (
            "RUBIES broad-component Gaussian FWHM was constrained below 2500 km/s; "
            "the high-width tail and any radius inference are censored."
        ),
        "legacy_reservoir_model_fwhm_kms_p16_median_p84": legacy_quantiles,
        "legacy_weighted_cdf_distance": legacy_ks,
        "compact_radius_descriptive_fit_au": fitted_radius,
        "compact_radius_bootstrap_68_percent_au": radius_interval.tolist(),
        "compact_model_fwhm_kms_p16_median_p84": compact_quantiles,
        "compact_weighted_cdf_distance": compact_ks,
        "inference_status": (
            "Descriptive scale calibration only; reverberation, profile modeling, "
            "or an uncensored width likelihood is needed for a physical posterior."
        ),
        "lifecycle_config": cfg.__dict__,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rubies_fwhm_confrontation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.45))
    axes[0].step(grid, obs_cdf, where="post", color="black", label="RUBIES LRDs")
    axes[0].plot(grid, legacy_cdf, color="#b75d47", ls="--", label="20–40 pc reservoir")
    axes[0].plot(grid, compact_cdf, color="#2267a5",
                 label=f"compact scale ({fitted_radius:.0f} AU)")
    axes[0].axvline(2500, color="0.5", ls=":", lw=1, label="fit ceiling")
    axes[0].set(xlabel=r"broad Balmer FWHM [km s$^{-1}$]", ylabel="cumulative fraction",
                xlim=(0, 4000), ylim=(0, 1.02))
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].semilogx(radii, model_medians, color="#2267a5")
    axes[1].axhline(observed_median, color="black", ls="--", label="RUBIES median")
    axes[1].axvline(fitted_radius, color="#2267a5", ls=":")
    axes[1].fill_betweenx([0, 4000], radius_interval[0], radius_interval[1],
                          color="#2267a5", alpha=0.12, label="bootstrap 68%")
    axes[1].set(xlabel="compact line radius [AU]", ylabel="model median FWHM [km s$^{-1}$]",
                ylim=(0, 4000))
    axes[1].xaxis.set_major_locator(FixedLocator([300, 1000, 3000]))
    axes[1].xaxis.set_major_formatter(FixedFormatter(["300", "1000", "3000"]))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Public RUBIES confrontation of the lifecycle broad-line scale")
    fig.tight_layout()
    fig.savefig(OUT / "rubies_fwhm_confrontation.pdf")
    fig.savefig(OUT / "rubies_fwhm_confrontation.png", dpi=220)
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
