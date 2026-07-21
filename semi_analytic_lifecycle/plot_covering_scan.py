#!/usr/bin/env python3
"""Figure 2: the joint scan of the covering law (fiducial flux-stack metric).

Reads ``covering_ratio_space_refit.json`` -- the epsilon=0.1 flux-stacked
line-ratio scan over (gamma, k) with the z<=7.5 demographic term (paper
Sec. 4.1 / Table 2) -- and writes ``coregulated_covering_scan.pdf`` / ``.png``:
  left  : a (gamma, k) heatmap of the joint objective (log color scale);
  right : the flux-stack spectral vs demographic components, coloured by gamma.

The scan itself is a full-grid computation (needs every configuration's forward
catalogue); this script only *plots* the shipped results, so it reproduces the
figure without the bulk catalogue grid.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
grid = json.loads((HERE / "covering_ratio_space_refit.json").read_text())["grid"]
lookup = {(r["gamma"], r["k"]): r for r in grid}

GAMMAS = [1.5, 2.0, 2.5, 3.0]          # populated range shown in the paper
KS = [0.5, 1.0, 1.5, 2.0]
J = np.full((len(KS), len(GAMMAS)), np.nan)
for i, k in enumerate(KS):
    for j, g in enumerate(GAMMAS):
        if (g, k) in lookup:
            J[i, j] = lookup[(g, k)]["joint_flux"]

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.6))

im = ax0.imshow(np.log10(J), origin="lower", aspect="auto", cmap="viridis_r",
                extent=[-0.5, len(GAMMAS) - 0.5, -0.5, len(KS) - 0.5])
ax0.set_xticks(range(len(GAMMAS))); ax0.set_xticklabels(GAMMAS)
ax0.set_yticks(range(len(KS))); ax0.set_yticklabels(KS)
ax0.set_xlabel(r"covering odds index $\gamma$")
ax0.set_ylabel(r"normalization $k$")
ax0.set_title(r"joint objective (flux-stack ratio + $z\leq7.5$ demographic $\chi^2$)")
for i in range(len(KS)):
    for j in range(len(GAMMAS)):
        if np.isfinite(J[i, j]):
            ax0.text(j, i, f"{J[i, j]:.0f}", ha="center", va="center",
                     color="white", fontsize=9)
fig.colorbar(im, ax=ax0, label=r"$\log_{10}$ joint objective")

sp = [r["chi2_ratio_flux_stack"] for r in grid]
dm = [r["demographic_chi2_z_le_7p5"] for r in grid]
gg = [r["gamma"] for r in grid]
sc = ax1.scatter(sp, dm, c=gg, cmap="plasma", s=60, edgecolor="k", lw=0.3)
ax1.set_xscale("log")
ax1.set_xlabel(r"flux-stack spectral $\chi^2$ (8 ratios)")
ax1.set_ylabel(r"demographic $\chi^2$ ($z\leq7.5$, 3 bins)")
best = lookup.get((2.0, 1.0))
if best:
    ax1.annotate(r"$(\gamma,k)=(2,1)$",
                 (best["chi2_ratio_flux_stack"], best["demographic_chi2_z_le_7p5"]),
                 xytext=(28, 22), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->"))
fig.colorbar(sc, ax=ax1, label=r"$\gamma$")

fig.tight_layout()
fig.savefig(HERE / "coregulated_covering_scan.pdf")
fig.savefig(HERE / "coregulated_covering_scan.png", dpi=150)
print("wrote coregulated_covering_scan.pdf/.png")
