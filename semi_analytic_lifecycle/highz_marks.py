#!/usr/bin/env python3
"""Channel-separated high-redshift predictions from the lifecycle catalogue."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "synthetic_catalog_early_visibility.csv"


def wquantile(x, w, q=(0.16, 0.5, 0.84)):
    order = np.argsort(x)
    x, w = np.asarray(x)[order], np.asarray(w)[order]
    cdf = (np.cumsum(w) - .5 * w) / np.sum(w)
    return np.interp(q, cdf, x).tolist()


def summary(values, weight):
    return {key: wquantile(values[key], weight) for key in
            ("log_lha", "log_lx", "lognh", "reff_pc", "fwhm_kms", "variability_rms")}


def main():
    rows = list(csv.DictReader(SOURCE.open()))
    output = {"scope": "selected synthetic catalogue; predictions are conditional on current proxy radiation and selection layers"}
    fig, ax = plt.subplots(2, 2, figsize=(7.5, 6.0))
    colours = {"secular_compaction": "#e47c22", "early_burst": "#7c4aa5"}
    for ztarget in (7.5, 9.5):
        snap = [r for r in rows if abs(float(r["z"]) - ztarget) < 0.1]
        zresult = {}
        for channel, colour in colours.items():
            part = [r for r in snap if r["entry_channel"] == channel]
            if not part:
                continue
            values = {key: np.array([float(r[key]) for r in part]) for key in
                      ("log_lha", "log_lx", "lognh", "reff_pc", "fwhm_kms", "variability_rms", "logmburst")}
            weight = np.array([float(r["weight_cMpc3"]) for r in part])
            zresult[channel] = {
                "weighted_fraction": float(weight.sum() / sum(float(r["weight_cMpc3"]) for r in snap)),
                "reservoir_visible_fraction": float(weight[values["logmburst"] > 4].sum() / weight.sum()),
                "marks_quantiles_16_50_84": summary(values, weight),
            }
            label = f"{channel.replace('_', ' ')}; z≈{ztarget:g}"
            ax[0, 0].scatter(values["log_lha"], values["log_lx"], s=6, alpha=.18, c=colour, label=label)
            ax[0, 1].scatter(values["lognh"], values["variability_rms"], s=6, alpha=.18, c=colour)
            ax[1, 0].scatter(values["reff_pc"], values["fwhm_kms"], s=6, alpha=.18, c=colour)
            ax[1, 1].hist(values["logmburst"], bins=np.linspace(0, 8.5, 30), weights=weight / weight.sum(),
                          histtype="step", lw=1.5, color=colour, label=label)
        output[f"z_{ztarget:g}"] = zresult
    ax[0, 0].set(xlabel=r"$\log L_{\mathrm{H}\alpha}$", ylabel=r"$\log L_X$")
    ax[0, 1].set(xlabel=r"$\log N_H$", ylabel="fractional variability")
    ax[1, 0].set(xlabel=r"$R_{\rm eff}$ [pc]", ylabel=r"H$\alpha$ FWHM [km s$^{-1}$]")
    ax[1, 1].set(xlabel=r"$\log M_{\rm burst}/M_\odot$", ylabel="weighted density")
    for a in ax.ravel():
        a.spines[["top", "right"]].set_visible(False)
    ax[0, 0].legend(frameon=False, fontsize=6)
    fig.suptitle("High-z lifecycle marks: predictions by formation channel", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(HERE / "highz_channel_marks.png", dpi=220)
    fig.savefig(HERE / "highz_channel_marks.pdf")
    (HERE / "highz_channel_marks.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
