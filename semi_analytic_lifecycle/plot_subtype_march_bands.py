#!/usr/bin/env python3
"""Companion-Letter Figure 1: the predicted subtype march with uncertainty bands.

Reads ``subtype_march_uncertainties.json`` -- the seed-averaged selected subtype
fractions with combined binomial and realization-to-realization uncertainties
(also tabulated as Letter Table 1) -- and writes
``subtype_march_with_bands.pdf`` / ``.png``.

The JSON maps each redshift to ``{subtype: [mean_fraction, uncertainty, N_eff]}``.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "subtype_march_uncertainties.json").read_text())

redshifts = sorted((float(z) for z in data), reverse=True)
subtypes = ["xLRD", "plusLRD", "minusLRD", "bLRD"]
colors = {"xLRD": "#b2182b", "plusLRD": "#ef8a62",
          "minusLRD": "#8073ac", "bLRD": "#2166ac"}


def series(subtype, idx):
    return np.array([data[f"{z:g}"][subtype][idx] for z in redshifts])


fig, ax = plt.subplots(figsize=(6.5, 4.2))
for st in subtypes:
    mean, err = series(st, 0), series(st, 1)
    ax.plot(redshifts, mean, "-o", color=colors[st], lw=2, label=st)
    ax.fill_between(redshifts, np.clip(mean - err, 0, 1), np.clip(mean + err, 0, 1),
                    color=colors[st], alpha=0.20)

ax.set_xlabel("redshift $z$")
ax.set_ylabel("selected subtype fraction")
ax.set_xlim(max(redshifts) + 0.3, min(redshifts) - 0.3)  # high-z on the left
ax.set_ylim(0, 1)
ax.legend(frameon=False, ncol=2)
ax.set_title("Predicted subtype march (seed-averaged, with uncertainty bands)")
fig.tight_layout()
fig.savefig(HERE / "subtype_march_with_bands.pdf")
fig.savefig(HERE / "subtype_march_with_bands.png", dpi=200)
print("wrote subtype_march_with_bands.pdf/.png")
