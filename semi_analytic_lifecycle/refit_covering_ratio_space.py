#!/usr/bin/env python3
"""Ratio-space refit of the covering law (referee revision, Tier 1).

The original (gamma, k) scan scored the physical photon partition against
the four stack-INVERTED dense fractions -- model-dependent quantities whose
0.05 floor hides an 11-sigma observable-space failure at the xLRD point
(REFEREE_REVIEW.md, I-1).  This script rescores every available scan
catalogue DIRECTLY against the eight stacked line ratios ([O III]5007/Hbeta
and [O III]/[O II] for the four subtypes), with:

- two stack constructions: median f_dense mapped through the soft two-zone
  emulator ("median-map", as in the closure paper), and luminosity-weighted
  stacking of per-source line yields ("flux-stack", closer to how real
  stacks average spectra);
- the independent He II 4686/Hbeta < 0.1 population gate;
- the demographic term restated on the z <= 7.5 bins only (review I-3).

Reference values: the imposed D^2 closure scores 5.77 on the same eight
ratios (median-map); the fitted (2,1) law scored 132 in the review check.

Outputs: covering_ratio_space_refit.json and a printed ranking.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "interception", HERE / "derive_interception_exponent.py")
interception = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(interception)

import sys
sys.path.insert(0, str(HERE))
from soft_two_zone_emulator import get_soft_two_zone_emulator

# Perez-Gonzalez et al. (2026) stacked observables per subtype:
# ([O III]/Hb, err, [O III]/[O II], err)
OBSERVED = {
    "xLRD": (1.10, 0.10, 14.8, 4.6),
    "plusLRD": (3.20, 0.10, 21.8, 6.6),
    "minusLRD": (4.40, 0.30, 23.6, 4.7),
    "bLRD": (4.90, 0.50, 17.5, 5.4),
}
HEII_GATE = 0.1                     # Sok et al. population upper envelope
SUBTYPES = ["xLRD", "plusLRD", "minusLRD", "bLRD"]
CANDIDATE = "D_dust_quenched_912"

GRID = [(g, k) for g in (1.5, 2.0, 2.5, 3.0, 3.5) for k in (0.5, 1.0, 1.5, 2.0)]


def suffix_for(gamma, k):
    if (gamma, k) == (2.0, 1.0):
        return "coregulated_covering_mid"
    return f"covscan_g{str(gamma).replace('.', 'p')}_k{str(k).replace('.', 'p')}"


def line_scalar(lines, key):
    return float(np.asarray(lines[key]).ravel()[0])


def ratios_from_fdense(emulator, f_dense):
    lines = emulator.predict(np.array([50000.0]), np.array([-2.1]),
                             np.array([float(f_dense)]), clip=False)
    hb = line_scalar(lines, "hbeta")
    oii = line_scalar(lines, "oii3726") + line_scalar(lines, "oii3729")
    return (line_scalar(lines, "oiii5007") / hb,
            line_scalar(lines, "oiii5007") / max(oii, 1.0e-30),
            line_scalar(lines, "heii4686") / hb)


def flux_stack_ratios(emulator, f_dense, lum_weights):
    """Luminosity-weighted stack: average line yields, then form ratios."""
    f = np.clip(np.asarray(f_dense, float), 0.0, 1.0)
    w = np.asarray(lum_weights, float)
    lines = emulator.predict(np.full_like(f, 50000.0), np.full_like(f, -2.1),
                             f, clip=False)
    def stack(key):
        return float(np.sum(w * np.asarray(lines[key])))
    hb = stack("hbeta")
    oii = stack("oii3726") + stack("oii3729")
    return (stack("oiii5007") / hb,
            stack("oiii5007") / max(oii, 1.0e-30),
            stack("heii4686") / hb)


def evaluate(suffix):
    csv_path = HERE / f"synthetic_catalog_{suffix}.csv"
    json_path = HERE / f"lifecycle_results_{suffix}.json"
    if not csv_path.exists() or not json_path.exists():
        return None
    interception.CATALOG = csv_path.name
    cat = interception.load()
    valid = (np.isfinite(cat["d"]) & (cat["d"] > 0)
             & np.isfinite(cat["t_escape"])
             & (cat["sfr"] > 0) & (cat["lbol"] > 0) & (cat["weight"] > 0))
    for key in cat:
        cat[key] = cat[key][valid]
    ws = cat["p_select"] * cat["weight"]
    ratio = interception.candidate_ratios(cat)[CANDIDATE]
    f_dense = ratio / (1.0 + ratio)
    window = (cat["z"] >= 4.5) & (cat["z"] <= 6.5) & (ws > 0)
    emulator = get_soft_two_zone_emulator()

    chi2_median, chi2_flux, heii_max = 0.0, 0.0, 0.0
    per_subtype = {}
    for subtype in SUBTYPES:
        mask = window & (cat["subtype"] == subtype)
        if mask.sum() == 0:
            return None
        med_f = interception.weighted_median(f_dense[mask], ws[mask])
        m_oiii_hb, m_oiii_oii, m_heii = ratios_from_fdense(emulator, med_f)
        # Luminosity weighting: stack contribution scales with the selected
        # weight times the ionizing output (proportional to L_bol).
        lw = ws[mask] * cat["lbol"][mask]
        lw = lw / lw.sum()
        s_oiii_hb, s_oiii_oii, s_heii = flux_stack_ratios(
            emulator, f_dense[mask], lw)
        obs = OBSERVED[subtype]
        chi2_median += (((m_oiii_hb - obs[0]) / obs[1]) ** 2
                        + ((m_oiii_oii - obs[2]) / obs[3]) ** 2)
        chi2_flux += (((s_oiii_hb - obs[0]) / obs[1]) ** 2
                      + ((s_oiii_oii - obs[2]) / obs[3]) ** 2)
        heii_max = max(heii_max, m_heii, s_heii)
        per_subtype[subtype] = {
            "median_f_dense": round(med_f, 4),
            "median_map_oiii_hb": round(m_oiii_hb, 3),
            "flux_stack_oiii_hb": round(s_oiii_hb, 3),
            "observed_oiii_hb": obs[0],
        }
    pulls = json.loads(json_path.read_text())["demographic_fit"]["pulls_sigma"]
    demog3 = float(sum(p ** 2 for p in pulls[:3]))
    return {
        "chi2_ratio_median_map": round(chi2_median, 2),
        "chi2_ratio_flux_stack": round(chi2_flux, 2),
        "max_heii_hbeta": round(heii_max, 4),
        "passes_heii_gate": bool(heii_max < HEII_GATE),
        "demographic_chi2_z_le_7p5": round(demog3, 2),
        "joint_flux": round(chi2_flux + demog3, 2),
        "per_subtype": per_subtype,
    }


def main():
    rows = []
    for gamma, k in GRID:
        result = evaluate(suffix_for(gamma, k))
        if result is None:
            continue
        rows.append({"gamma": gamma, "k": k, **result})
    rows.sort(key=lambda r: r["joint_flux"])

    payload = {
        "objective": "chi2 over 8 stacked line ratios (flux-stack) "
                     "+ demographic chi2 on z<=7.5 bins; He II gate reported",
        "closure_reference_median_map": 5.77,
        "grid": rows,
        "best": rows[0] if rows else None,
    }
    (HERE / "covering_ratio_space_refit.json").write_text(
        json.dumps(payload, indent=2) + "\n")

    print(f"{'g':>4} {'k':>4} {'chi2_med':>9} {'chi2_flux':>10} "
          f"{'demog3':>7} {'joint':>7} {'HeII_ok':>8}  xLRD OIII/Hb (med/flux vs 1.10)")
    for r in rows:
        x = r["per_subtype"]["xLRD"]
        print(f"{r['gamma']:>4} {r['k']:>4} {r['chi2_ratio_median_map']:>9} "
              f"{r['chi2_ratio_flux_stack']:>10} "
              f"{r['demographic_chi2_z_le_7p5']:>7} {r['joint_flux']:>7} "
              f"{str(r['passes_heii_gate']):>8}  "
              f"{x['median_map_oiii_hb']}/{x['flux_stack_oiii_hb']}")


if __name__ == "__main__":
    main()
