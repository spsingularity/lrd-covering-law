#!/usr/bin/env python3
"""Honest first confrontation of lifecycle marks with public JADES DR4 data.

The public DR4 spectra do not identify the final visually screened LRD sample.
Consequently this is a UV/redshift-matched *control* comparison, not an LRD
likelihood or a model fit.  It establishes the interfaces, reports exactly
which marks can be tested now, and prevents parent-galaxy data being silently
treated as LRD data.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM


HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parent / "catalog_likelihood" / "jades_dr4_spectral_likelihood_table.csv"
MODEL = HERE / "synthetic_catalog_early_visibility.csv"
OUT_JSON = HERE / "marks_confrontation.json"
OUT_PNG = HERE / "marks_confrontation.png"
OUT_PDF = HERE / "marks_confrontation.pdf"
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)


def weighted_quantile(value, weight, quantiles=(0.16, 0.5, 0.84)):
    value, weight = np.asarray(value, float), np.asarray(weight, float)
    ok = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    value, weight = value[ok], weight[ok]
    order = np.argsort(value)
    value, weight = value[order], weight[order]
    cdf = (np.cumsum(weight) - 0.5 * weight) / np.sum(weight)
    return np.interp(quantiles, cdf, value).tolist()


def load_public_control():
    rows = list(csv.DictReader(PUBLIC.open()))
    z = np.array([float(row["z_spec"]) for row in rows])
    muv = np.array([float(row["muv"]) for row in rows])
    flux = np.array([float(row["HBaA_6563_flux"]) for row in rows])
    error = np.array([float(row["HBaA_6563_error"]) for row in rows])
    detected = np.array([int(row["HBaA_6563_detected"]) == 1 for row in rows])
    # Match the usable model snapshot and broad model UV support.  Fluxes in
    # this DR4 table are explicitly in 1e-20 erg s^-1 cm^-2.
    selected = ((z >= 5.0) & (z < 6.0) & (muv >= -20.5) & (muv <= -18.0)
                & np.isfinite(flux) & np.isfinite(error) & (error > 0))
    luminosity = 4.0 * np.pi * COSMO.luminosity_distance(z[selected]).to_value("cm")**2
    log_lha = np.log10(luminosity * flux[selected] * 1e-20)
    return {
        "z": z[selected], "muv": muv[selected], "log_lha": log_lha,
        "detected": detected[selected], "n_selected": int(np.sum(selected)),
        "n_detected": int(np.sum(detected[selected])),
    }


def load_model():
    rows = list(csv.DictReader(MODEL.open()))
    # One model snapshot avoids treating time-adjacent snapshots as
    # independent galaxies.  z=5.51 is the closest to the DR4 control slice.
    selected = [row for row in rows if abs(float(row["z"]) - 5.5) < 0.1
                and -20.5 <= float(row["muv"]) <= -18.0]
    values = {key: np.array([float(row[key]) for row in selected])
              for key in ("log_lha", "log_lx", "variability_rms", "logmburst", "weight_cMpc3")}
    values["channel"] = np.array([row["entry_channel"] for row in selected])
    values["muv"] = np.array([float(row["muv"]) for row in selected])
    return values


def main():
    public, model = load_public_control(), load_model()
    detected_lha = public["log_lha"][public["detected"]]
    q_public = np.quantile(detected_lha, [0.16, 0.5, 0.84]).tolist()
    q_model = weighted_quantile(model["log_lha"], model["weight_cMpc3"])
    channels = {}
    for channel in ("secular_compaction", "early_burst"):
        m = model["channel"] == channel
        channels[channel] = {
            "weighted_fraction": float(np.sum(model["weight_cMpc3"][m]) /
                                       np.sum(model["weight_cMpc3"])),
            "log_lha_quantiles": weighted_quantile(model["log_lha"][m], model["weight_cMpc3"][m]),
            "log_lx_quantiles": weighted_quantile(model["log_lx"][m], model["weight_cMpc3"][m]),
            "variability_quantiles": weighted_quantile(model["variability_rms"][m], model["weight_cMpc3"][m]),
            "reservoir_visible_fraction": float(np.sum(model["weight_cMpc3"][m & (model["logmburst"] > 4)]) /
                                                 np.sum(model["weight_cMpc3"][m])),
        }
    summary = {
        "scope": "UV/redshift-matched public JADES DR4 control comparison, not an LRD-selected likelihood",
        "selection": {"z": "5.0 <= z_spec < 6.0", "muv": "-20.5 <= MUV <= -18.0"},
        "public_halpha": {
            "n_control": public["n_selected"], "n_detected": public["n_detected"],
            "log_lha_detected_quantiles": q_public,
            "flux_unit": "1e-20 erg s^-1 cm^-2",
        },
        "model_halpha": {
            "n_synthetic_support": int(len(model["log_lha"])),
            "log_lha_weighted_quantiles": q_model,
            "median_offset_model_minus_control_dex": float(q_model[1] - q_public[1]),
        },
        "channel_predictions": channels,
        "not_yet_testable": [
            "LRD-specific H-alpha distribution: final LRD IDs and visual decisions are unreleased",
            "X-ray weakness: no matched X-ray fluxes/upper limits supplied for the final LRD sample",
            "variability: no matched time-domain measurements supplied for the final LRD sample",
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    bins = np.linspace(40.5, 44.0, 28)
    ax[0].hist(detected_lha, bins=bins, density=True, histtype="step", lw=1.8,
               color="black", label="public DR4 control")
    ax[0].hist(model["log_lha"], bins=bins, weights=model["weight_cMpc3"], density=True,
               histtype="step", lw=1.8, color="#2864a5", label="selected lifecycle model")
    ax[0].set(xlabel=r"$\log L_{\mathrm{H}\alpha}$ [erg s$^{-1}$]", ylabel="density")
    ax[0].legend(frameon=False, fontsize=7)
    colours = {"secular_compaction": "#e47c22", "early_burst": "#7c4aa5"}
    for channel, colour in colours.items():
        m = model["channel"] == channel
        ax[1].scatter(model["log_lx"][m], model["variability_rms"][m],
                      c=colour, s=8, alpha=0.25, linewidth=0, label=channel.replace("_", " "))
    ax[1].set(xlabel=r"predicted $\log L_X$ [erg s$^{-1}$]", ylabel="predicted fractional variability")
    ax[1].legend(frameon=False, fontsize=7)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Lifecycle marks: public-control confrontation and channel predictions", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=220)
    fig.savefig(OUT_PDF)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
