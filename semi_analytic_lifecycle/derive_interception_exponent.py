#!/usr/bin/env python3
"""Branch-2 test of the eta=2 derivation: emergent interception exponent.

The quadratic closure imposes Q_dense/Q_diffuse = D^2 at the photon-partition
level.  The exponent derivation in the manuscript shows the covering
odds inside the lifecycle carry only ~0.1 power of D, so a covering-based
second factor (branch 1) is not present in the current implementation.  This
script tests branch 2: build the dense/diffuse ionizing-photon competition
from quantities that exist *upstream* of the closure, and measure the
exponent that emerges, rather than the one that is imposed.

Physical partition candidates, per source:

  Q_e  = 0.5 * L_bol / (3.26 Ryd)          (the lifecycle's own conversion)
  Q_*  = QSTAR_PER_SFR * uv_youth * SFR    (stellar ionizing output)
  t    = visibility_qh_escape_fraction     (porous-envelope Q(H) escape)

  Model A: R = (1 - t) / t                  pure interception odds
  Model B: R = (1 - t) * Q_e / Q_*          stellar-dominated diffuse zone
  Model C: R = (1 - t) * Q_e / (t * Q_e + Q_*)
                                            full competition: escaped nuclear
                                            photons also feed the diffuse gas
  Model D: as B/C but with the stellar ionizing supply quenched by the host
           attenuation reconstructed from the model's own lambda^-1.2 dust
           law (the same dust that builds D's denominator)

The script also computes the *required* Lemma-2 law: the per-subtype
correction factor that the best physical candidate would need to reproduce
the stack-inverted fractions, regressed against median D.  If the required
correction is close to a single power law in D, it specifies exactly the
covering--dominance co-regulation the lifecycle currently lacks.

Every model is regressed as log R = eta_emergent * log D + const on the same
sources the closure uses, and each physical partition is pushed through
f_dense = R/(1+R) to confront the four stack-inverted dense fractions with
zero closure imposition.  The stellar conversion constant shifts only the
intercept, never the measured exponent; a factor-of-three bracket is still
reported for the stack confrontation, which does depend on it.

Outputs: interception_exponent_summary.json, interception_exponent.pdf/.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CATALOG = "synthetic_catalog_physical_visibility_final.csv"

RYDBERG_ERG = 2.1798723611030e-11
IONIZING_BOLOMETRIC_FRACTION = 0.5      # lrd_lifecycle.py config value
MEAN_IONIZING_ENERGY_RYD = 3.26         # lrd_lifecycle.py config value
QSTAR_PER_SFR = 1.0e53                  # photons/s per Msun/yr, central value
QSTAR_BRACKET = (1.0 / 3.0, 3.0)        # factor-of-three systematic bracket

STACK_INVERTED = {  # dense Q(H) fractions inverted from the public stacks
    "xLRD": (0.981, 0.978, 0.983),
    "plusLRD": (0.901, 0.884, 0.911),
    "minusLRD": (0.738, 0.625, 0.807),
    "bLRD": (0.086, 0.000, 0.658),
}
SYSTEMATIC_FLOOR = 0.05                 # the closure paper's declared floor
SUBTYPES = ["xLRD", "plusLRD", "minusLRD", "bLRD"]


H_PLANCK = 6.62607015e-27
K_BOLTZMANN = 1.380649e-16
C_LIGHT = 2.99792458e10
SIGMA_SB = 5.670374419e-5


def blackbody_lnu_per_lbol(wavelength_angstrom, temperature_k):
    nu = C_LIGHT / (wavelength_angstrom * 1.0e-8)
    x = H_PLANCK * nu / (K_BOLTZMANN * temperature_k)
    bnu = 2.0 * H_PLANCK * nu**3 / C_LIGHT**2 / np.expm1(np.clip(x, 0.0, 700.0))
    return np.pi * bnu / (SIGMA_SB * temperature_k**4)


def load() -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(HERE / CATALOG)))
    def col(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in rows])
    cat = {
        "z": col("z"),
        "subtype": np.array([r["visibility_subtype"] for r in rows]),
        "d": col("visibility_core_to_host_5100"),
        "t_escape": col("visibility_qh_escape_fraction"),
        "lbol": 10.0 ** col("log_lbol"),
        "lthermal": 10.0 ** col("log_lthermal"),
        "sfr": col("sfr_msunyr"),
        "p_select": col("p_select"),
        "weight": col("weight_cMpc3"),
    }
    # Reconstruct the host attenuation from the continuum layer's own
    # equations: host_5100 = D^-1 * thermal_5100 and the intrinsic stellar
    # 5100 A output, then invert the lambda^-1.2 attenuation law.
    uv_youth = np.clip(((1.0 + cat["z"]) / 6.0) ** 1.5, 0.7, 2.5)
    star_5100_intrinsic = 8.0e27 * uv_youth * cat["sfr"]
    thermal_5100 = cat["lthermal"] * blackbody_lnu_per_lbol(5100.0, 4050.0)
    host_5100 = np.divide(thermal_5100, cat["d"],
                          out=np.zeros_like(cat["d"]), where=cat["d"] > 0)
    atten_5100 = np.clip(
        host_5100 / np.maximum(star_5100_intrinsic, 1.0e-30), 1.0e-12, 1.0)
    a_5100 = -2.5 * np.log10(atten_5100)
    cat["a_2500_mag"] = a_5100 / (5100.0 / 2500.0) ** -1.2
    cat["a_912_mag"] = cat["a_2500_mag"] * (912.0 / 2500.0) ** -1.2
    return cat


def photon_budgets(cat: dict[str, np.ndarray], qstar_factor: float = 1.0):
    q_engine = (IONIZING_BOLOMETRIC_FRACTION * cat["lbol"]
                / (MEAN_IONIZING_ENERGY_RYD * RYDBERG_ERG))
    # Same youth factor the continuum layer applies to the 2500 A output.
    uv_youth = np.clip(((1.0 + cat["z"]) / 6.0) ** 1.5, 0.7, 2.5)
    q_star = QSTAR_PER_SFR * qstar_factor * uv_youth * cat["sfr"]
    return q_engine, q_star


def candidate_ratios(cat, qstar_factor: float = 1.0):
    q_e, q_s = photon_budgets(cat, qstar_factor)
    t = np.clip(cat["t_escape"], 1.0e-12, 1.0 - 1.0e-12)
    intercepted = (1.0 - t) * q_e
    q_s_dust_912 = q_s * 10.0 ** (-0.4 * cat["a_912_mag"])
    q_s_dust_2500 = q_s * 10.0 ** (-0.4 * cat["a_2500_mag"])
    return {
        "A_interception_odds": intercepted / (t * q_e),
        "B_stellar_diffuse": intercepted / q_s,
        "C_full_competition": intercepted / (t * q_e + q_s),
        "D_dust_quenched_912": intercepted / (t * q_e + q_s_dust_912),
        "D_dust_quenched_2500": intercepted / q_s_dust_2500,
    }


def required_lemma2_law(cat, ratios, weights, window=(4.5, 6.5),
                        reference="B_stellar_diffuse"):
    """Fit the covering law the physical partition is missing.

    For each subtype, the correction factor is R_required / R_physical, with
    R_required = f_inv/(1-f_inv) from the stack inversion.  Regressing its
    logarithm on the subtype's median log D yields the power of D that a
    revised interception (Lemma 2) must supply.
    """
    in_window = (cat["z"] >= window[0]) & (cat["z"] <= window[1])
    ratio = ratios[reference]
    log_corrections, log_d_medians = [], []
    per_subtype = {}
    for subtype in SUBTYPES:
        mask = in_window & (cat["subtype"] == subtype) & (weights > 0)
        if mask.sum() == 0:
            continue
        med_r = weighted_median(ratio[mask], weights[mask])
        med_d = weighted_median(cat["d"][mask], weights[mask])
        center = STACK_INVERTED[subtype][0]
        required = center / max(1.0 - center, 1.0e-6)
        correction = required / med_r
        per_subtype[subtype] = {
            "median_d": round(med_d, 3),
            "physical_odds": round(med_r, 3),
            "required_odds": round(required, 3),
            "correction_factor": round(correction, 3),
        }
        log_corrections.append(np.log10(correction))
        log_d_medians.append(np.log10(med_d))
    slope, intercept = np.polyfit(log_d_medians, log_corrections, 1)
    return {
        "reference_model": reference,
        "per_subtype": per_subtype,
        "required_extra_power_of_d": round(float(slope), 3),
        "required_normalization_log10": round(float(intercept), 3),
    }


def weighted_fit(log_d, log_r, weights):
    w = weights / weights.sum()
    mx, my = np.sum(w * log_d), np.sum(w * log_r)
    cov = np.sum(w * (log_d - mx) * (log_r - my))
    var = np.sum(w * (log_d - mx) ** 2)
    slope = cov / var
    corr = cov / np.sqrt(var * np.sum(w * (log_r - my) ** 2))
    return {"eta_emergent": float(slope),
            "intercept": float(my - slope * mx),
            "corr": float(corr)}


def weighted_median(values, weights):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, 0.5 * cum[-1])])


def stack_confrontation(cat, ratios, weights, window=(4.5, 6.5)):
    in_window = (cat["z"] >= window[0]) & (cat["z"] <= window[1])
    out = {}
    for name, ratio in ratios.items():
        f_dense = ratio / (1.0 + ratio)
        medians, chi2 = {}, 0.0
        for subtype in SUBTYPES:
            mask = in_window & (cat["subtype"] == subtype) & (weights > 0)
            if mask.sum() == 0:
                medians[subtype] = np.nan
                continue
            med = weighted_median(f_dense[mask], weights[mask])
            medians[subtype] = round(med, 4)
            center, p16, p84 = STACK_INVERTED[subtype]
            sigma = np.hypot(0.5 * (p84 - p16), SYSTEMATIC_FLOOR)
            chi2 += ((med - center) / sigma) ** 2
        out[name] = {"median_f_dense": medians, "chi2_vs_stacks": round(chi2, 2)}
    return out


def main() -> None:
    cat = load()
    valid = (
        np.isfinite(cat["d"]) & (cat["d"] > 0)
        & np.isfinite(cat["t_escape"])
        & (cat["sfr"] > 0) & (cat["lbol"] > 0) & (cat["weight"] > 0)
    )
    for key in cat:
        cat[key] = cat[key][valid]
    weights = cat["p_select"] * cat["weight"]
    log_d = np.log10(cat["d"])

    ratios = candidate_ratios(cat)
    window = (cat["z"] >= 4.5) & (cat["z"] <= 6.5) & (weights > 0)

    regressions = {}
    for name, ratio in ratios.items():
        log_r = np.log10(np.clip(ratio, 1.0e-30, None))
        regressions[name] = {
            "global": weighted_fit(log_d, log_r, weights),
            "z45_65": weighted_fit(log_d[window], log_r[window], weights[window]),
        }

    confrontation = stack_confrontation(cat, ratios, weights)
    lemma2_requirement = required_lemma2_law(cat, ratios, weights)
    bracket = {}
    for factor in QSTAR_BRACKET:
        bracketed = stack_confrontation(cat, candidate_ratios(cat, factor), weights)
        bracket[f"qstar_x{factor:.2f}"] = {
            name: result["chi2_vs_stacks"] for name, result in bracketed.items()
        }
    # Reference: what the imposed quadratic closure scores under the identical
    # medians-vs-stacks metric, so the physical candidates are compared fairly.
    closure_ref = stack_confrontation(
        cat, {"imposed_D2_closure": cat["d"] ** 2}, weights)

    payload = {
        "catalog": CATALOG,
        "n_sources": int(valid.sum()),
        "constants": {
            "ionizing_bolometric_fraction": IONIZING_BOLOMETRIC_FRACTION,
            "mean_ionizing_energy_ryd": MEAN_IONIZING_ENERGY_RYD,
            "qstar_per_sfr": QSTAR_PER_SFR,
        },
        "regressions": regressions,
        "stack_confrontation": confrontation,
        "closure_reference": closure_ref,
        "required_lemma2_law": lemma2_requirement,
        "qstar_bracket_chi2": bracket,
        "notes": [
            "eta_emergent is the weighted slope of log R on log D; the imposed",
            "closure would give exactly 2.0 by construction and is excluded here.",
            "The stellar conversion constant cannot change eta_emergent, only",
            "the stack-confrontation chi2, for which the factor-3 bracket is",
            "reported.",
        ],
    }
    (HERE / "interception_exponent_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n")

    plotted = ["A_interception_odds", "B_stellar_diffuse",
               "C_full_competition", "D_dust_quenched_912"]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.0), sharey=True)
    labels = {"A_interception_odds": "A: $(1-t)/t$",
              "B_stellar_diffuse": "B: $(1-t)Q_e/Q_*$",
              "C_full_competition": "C: $(1-t)Q_e/(tQ_e+Q_*)$",
              "D_dust_quenched_912": "D: dust-quenched diffuse"}
    for ax, name in zip(axes, plotted):
        ratio = ratios[name]
        log_r = np.log10(np.clip(ratio, 1.0e-30, None))
        ax.scatter(log_d[window], log_r[window], s=4, alpha=0.2, color="#4C78A8")
        fit = regressions[name]["z45_65"]
        grid = np.linspace(log_d[window].min(), log_d[window].max(), 50)
        ax.plot(grid, fit["eta_emergent"] * grid + fit["intercept"],
                color="#8B0000", lw=2,
                label=f"$\\eta_{{emergent}}$ = {fit['eta_emergent']:.2f}")
        ax.plot(grid, 2.0 * grid + np.median(log_r[window] - 2.0 * log_d[window]),
                color="0.4", ls="--", lw=1.2, label="imposed $\\eta=2$")
        ax.set_xlabel(r"$\log_{10} D$")
        ax.set_title(labels[name], fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(r"$\log_{10} (Q_{\rm dense}/Q_{\rm diffuse})$")
    fig.suptitle("Emergent interception exponent, $4.5 \\leq z \\leq 6.5$", y=1.0)
    fig.tight_layout()
    fig.savefig(HERE / "interception_exponent.pdf")
    fig.savefig(HERE / "interception_exponent.png", dpi=200)
    plt.close(fig)

    print(json.dumps({
        "eta_emergent_z45_65": {
            name: round(r["z45_65"]["eta_emergent"], 3)
            for name, r in regressions.items()},
        "corr_z45_65": {
            name: round(r["z45_65"]["corr"], 3)
            for name, r in regressions.items()},
        "stack_chi2": {
            name: r["chi2_vs_stacks"] for name, r in confrontation.items()},
        "closure_reference_chi2":
            closure_ref["imposed_D2_closure"]["chi2_vs_stacks"],
        "stack_medians": {
            name: r["median_f_dense"] for name, r in confrontation.items()},
        "required_lemma2_law": lemma2_requirement,
        "qstar_bracket_chi2": bracket,
    }, indent=2))


if __name__ == "__main__":
    main()
