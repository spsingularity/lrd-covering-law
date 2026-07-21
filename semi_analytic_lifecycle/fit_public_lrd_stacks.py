"""Invert the public four-stack LRD constraints through the two-zone model.

The fit uses [O III]/Hbeta and [O III]/[O II] to infer the dense intercepted
Q(H) fraction and a discrete diffuse ionization parameter.  Balmer ratios are
then a held-out transfer test.  A one-parameter smooth screen is compared with
line-specific effective transfer boosts so that dust attenuation is not
silently forced to explain optically thick Balmer radiative transfer.
"""
from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import chi2 as chi2_distribution

from run_cloudy_lrd_pilot import diagnostics
from two_zone_cloudy import load_model, mix_yields


HERE = Path(__file__).resolve().parent
DATA = HERE / "public_lrd_constraints" / "perez_gonzalez26_lrd_stack_constraints.csv"
OUT = HERE / "public_lrd_constraints"
WAVELENGTH_ANGSTROM = {"halpha": 6562.80, "hbeta": 4861.32, "hgamma": 4340.46}


def load_stacks():
    with DATA.open() as handle:
        rows = list(csv.DictReader(handle))
    numeric = [key for key in rows[0] if key not in {"subtype", "source"}]
    return [{key: (float(value) if key in numeric else value)
             for key, value in row.items()} for row in rows]


def model_names(temperature_k):
    stem = f"soft_diffuse_{temperature_k // 1000}k"
    return (
        (stem, -2.5),
        (f"{stem}_u_m2p0", -2.0),
        (f"{stem}_u_m1p5", -1.5),
    )


@lru_cache(maxsize=None)
def diffuse_anchor_grid(temperature_k):
    anchors = [(anchor_logu, load_model(name)["erg_per_ionizing_photon"])
               for name, anchor_logu in model_names(temperature_k)]
    anchors.sort(key=lambda item: item[0])
    grid = np.array([item[0] for item in anchors])
    line_grids = {
        line: np.log10([max(item[1][line], 1.0e-99) for item in anchors])
        for line in anchors[0][1]
    }
    return grid, line_grids


def interpolate_diffuse_yields(temperature_k, logu):
    """Log-linearly interpolate positive Cloudy yields across the log-U grid."""
    grid, line_grids = diffuse_anchor_grid(temperature_k)
    if not grid[0] <= logu <= grid[-1]:
        raise ValueError("logu lies outside the Cloudy anchor grid")
    return {
        line: float(10.0 ** np.interp(logu, grid, values))
        for line, values in line_grids.items()
    }


def ratio_chi2(diagnostic, observed):
    return (
        (diagnostic["oiii_hbeta"] - observed["oiii5007_hbeta"])**2
        / observed["oiii5007_hbeta_error"]**2
        + (diagnostic["oiii5007_oii3727"] - observed["oiii5007_oii3727"])**2
        / observed["oiii5007_oii3727_error"]**2
    )


def fit_dense_fraction(dense_yields, diffuse_yields, observed):
    def objective(fraction):
        values = diagnostics(mix_yields(dense_yields, diffuse_yields, fraction))
        return ratio_chi2(values, observed)

    solution = minimize_scalar(
        objective, bounds=(0.0, 0.999999999), method="bounded",
        options={"xatol": 1.0e-11, "maxiter": 1000},
    )
    fraction = float(solution.x)
    values = diagnostics(mix_yields(dense_yields, diffuse_yields, fraction))
    return fraction, values, float(solution.fun)


def fit_two_zone_ratios(temperature_k, dense_yields, observed, initial=None):
    """Jointly infer dense Q(H) fraction and continuous interpolated log U."""
    def residual(parameters):
        fraction, logu = parameters
        diffuse = interpolate_diffuse_yields(temperature_k, logu)
        values = diagnostics(mix_yields(dense_yields, diffuse, fraction))
        return np.array([
            (values["oiii_hbeta"] - observed["oiii5007_hbeta"])
            / observed["oiii5007_hbeta_error"],
            (values["oiii5007_oii3727"] - observed["oiii5007_oii3727"])
            / observed["oiii5007_oii3727_error"],
        ])

    starts = ([initial] if initial is not None else
              [(fraction, logu) for fraction in (0.2, 0.8, 0.95, 0.99)
               for logu in (-2.45, -2.25, -2.05, -1.75)])
    solutions = []
    for fraction, logu in starts:
            solution = least_squares(
                residual, x0=(fraction, logu),
                bounds=((0.0, -2.5), (0.999999999, -1.5)),
                xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12,
                max_nfev=2000,
            )
            solutions.append(solution)
    solution = min(solutions, key=lambda item: float(np.sum(item.fun**2)))
    fraction, logu = map(float, solution.x)
    diffuse = interpolate_diffuse_yields(temperature_k, logu)
    values = diagnostics(mix_yields(dense_yields, diffuse, fraction))
    return fraction, logu, values, float(np.sum(solution.fun**2))


def measurement_uncertainties(temperature_k, dense_yields, observed,
                              central_fraction, central_logu, draws=400):
    """Propagate reported stack-ratio errors, excluding model systematics."""
    seed = 20260717 + sum(map(ord, observed["subtype"]))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        realization = dict(observed)
        for key in ("oiii5007_hbeta", "oiii5007_oii3727"):
            value = rng.normal(observed[key], observed[f"{key}_error"])
            realization[key] = max(value, 1.0e-6)
        fraction, logu, values, _ = fit_two_zone_ratios(
            temperature_k, dense_yields, realization,
            initial=(central_fraction, central_logu),
        )
        samples.append((fraction, logu, values["heii4686_hbeta"]))
    samples = np.asarray(samples)
    return {
        "dense_qh_fraction_measurement_p16_p50_p84":
            np.quantile(samples[:, 0], [0.16, 0.5, 0.84]).tolist(),
        "diffuse_logu_measurement_p16_p50_p84":
            np.quantile(samples[:, 1], [0.16, 0.5, 0.84]).tolist(),
        "heii4686_hbeta_measurement_p16_p50_p84":
            np.quantile(samples[:, 2], [0.16, 0.5, 0.84]).tolist(),
        "measurement_monte_carlo_draws": draws,
        "systematics_included": False,
    }


def screen_differentials(power=1.2):
    tau = {line: (wavelength / 5500.0)**(-power)
           for line, wavelength in WAVELENGTH_ANGSTROM.items()}
    return {
        "halpha_hbeta": tau["hbeta"] - tau["halpha"],
        "hbeta_hgamma": tau["hgamma"] - tau["hbeta"],
    }


def fit_smooth_screen(intrinsic, observed, power=1.2):
    """Fit a non-negative power-law foreground optical depth in log ratios."""
    delta = screen_differentials(power)
    keys = ("halpha_hbeta", "hbeta_hgamma")
    errors = {
        "halpha_hbeta": observed["halpha_hbeta_error"],
        "hbeta_hgamma": observed["hbeta_hgamma_error"],
    }
    y = {key: math.log(observed[key] / intrinsic[key]) for key in keys}
    sigma_log = {key: errors[key] / observed[key] for key in keys}
    numerator = sum(delta[key] * y[key] / sigma_log[key]**2 for key in keys)
    denominator = sum(delta[key]**2 / sigma_log[key]**2 for key in keys)
    tau_v = max(0.0, numerator / denominator)
    predicted = {key: intrinsic[key] * math.exp(tau_v * delta[key]) for key in keys}
    chi2 = sum(((predicted[key] - observed[key]) / errors[key])**2 for key in keys)
    return {
        "optical_depth_v": tau_v,
        "effective_a_v_magnitude": 1.086 * tau_v,
        "power_law_index": power,
        "predicted_halpha_hbeta": predicted["halpha_hbeta"],
        "predicted_hbeta_hgamma": predicted["hbeta_hgamma"],
        "chi2_two_ratios_one_parameter": chi2,
    }


def fit_temperature(temperature_k, stacks):
    dense_name = f"soft_dense_{temperature_k // 1000}k"
    dense = load_model(dense_name)["erg_per_ionizing_photon"]
    rows = []
    for stack in stacks:
        fraction, logu, values, chi2 = fit_two_zone_ratios(
            temperature_k, dense, stack
        )
        intrinsic = {
            "halpha_hbeta": values["halpha_hbeta"],
            "hbeta_hgamma": values["hbeta_hgamma"],
        }
        screen = fit_smooth_screen(intrinsic, stack)
        rows.append({
            "subtype": stack["subtype"],
            "temperature_basis_k": temperature_k,
            "dense_model": dense_name,
            "diffuse_model": "log-linear interpolation of thermal Cloudy anchors",
            "diffuse_logu": logu,
            "dense_qh_fraction": fraction,
            "line_ratio_chi2": chi2,
            "predicted_oiii5007_hbeta": values["oiii_hbeta"],
            "predicted_oiii5007_oii3727": values["oiii5007_oii3727"],
            "predicted_heii4686_hbeta": values["heii4686_hbeta"],
            "intrinsic_halpha_hbeta": intrinsic["halpha_hbeta"],
            "intrinsic_hbeta_hgamma": intrinsic["hbeta_hgamma"],
            "effective_halpha_hbeta_transfer_boost":
                stack["halpha_hbeta"] / intrinsic["halpha_hbeta"],
            "effective_hbeta_hgamma_transfer_boost":
                stack["hbeta_hgamma"] / intrinsic["hbeta_hgamma"],
            **{f"smooth_screen_{key}": value for key, value in screen.items()},
        })
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    stacks = load_stacks()
    fits = {temperature: fit_temperature(temperature, stacks)
            for temperature in (50000, 70000, 100000)}
    comparisons = {}
    for temperature, rows in fits.items():
        comparisons[str(temperature)] = {
            "line_ratio_chi2_sum": sum(row["line_ratio_chi2"] for row in rows),
            "maximum_heii4686_hbeta": max(row["predicted_heii4686_hbeta"] for row in rows),
            "all_fits_below_typical_heii_upper_0p1": all(
                row["predicted_heii4686_hbeta"] <= 0.1 for row in rows
            ),
            "smooth_screen_chi2_sum": sum(
                row["smooth_screen_chi2_two_ratios_one_parameter"] for row in rows
            ),
        }
    compatible = [temperature for temperature, rows in fits.items()
                  if all(row["predicted_heii4686_hbeta"] <= 0.1 for row in rows)]
    pool = compatible or list(fits)
    preferred_temperature = min(
        pool, key=lambda temperature: comparisons[str(temperature)]["line_ratio_chi2_sum"]
    )
    preferred = fits[preferred_temperature]
    preferred_dense = load_model(f"soft_dense_{preferred_temperature // 1000}k")[
        "erg_per_ionizing_photon"
    ]
    stacks_by_name = {stack["subtype"]: stack for stack in stacks}
    for row in preferred:
        row.update(measurement_uncertainties(
            preferred_temperature, preferred_dense, stacks_by_name[row["subtype"]],
            row["dense_qh_fraction"], row["diffuse_logu"],
        ))

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "public_lrd_stack_two_zone_fits.csv",
              [row for temperature in fits for row in fits[temperature]])
    summary = {
        "method": (
            "Two-zone thermal Cloudy inversion of [O III]/Hbeta and [O III]/[O II]. "
            "Each stack has a dense intercepted-Q(H) fraction and a diffuse log U "
            "interpolated within the thermal anchor grid. Two ratios determine two "
            "parameters, so the inversion itself is not a goodness-of-fit test."
        ),
        "temperature_basis_warning": (
            "The blackbody temperature is a hardness basis, not a literal source temperature."
        ),
        "heii_constraint_status": (
            "Sok et al. 2026 report HeII4686/Hbeta <= about 0.1 for most of the "
            "grating sample; it is used as a population compatibility gate, not "
            "as four independent subtype measurements."
        ),
        "temperature_comparison": comparisons,
        "preferred_temperature_basis_k": preferred_temperature,
        "preferred_fits": preferred,
        "balmer_smooth_screen_total_chi2": sum(
            row["smooth_screen_chi2_two_ratios_one_parameter"] for row in preferred
        ),
        "balmer_smooth_screen_nominal_dof": len(preferred),
    }
    summary["balmer_smooth_screen_p_value"] = float(chi2_distribution.sf(
        summary["balmer_smooth_screen_total_chi2"],
        summary["balmer_smooth_screen_nominal_dof"],
    ))
    summary["balmer_result"] = (
        "The smooth-screen model is not rejected by the present Hgamma uncertainties, "
        "but it requires A_V about 2.1--3.7 mag toward the Balmer-emitting region. "
        "Line transfer and differential broad/narrow attenuation remain alternatives, "
        "not demonstrated necessities."
    )
    (OUT / "public_lrd_stack_two_zone_fit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    colours = {"xLRD": "#8b2d3b", "plusLRD": "#d47934",
               "minusLRD": "#4c8b6b", "bLRD": "#2766a0"}
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.45))
    dense = load_model(f"soft_dense_{preferred_temperature // 1000}k")[
        "erg_per_ionizing_photon"
    ]
    fractions = np.linspace(0.0, 0.999, 500)
    for diffuse_name, logu in model_names(preferred_temperature):
        if logu > -1.9:
            continue
        diffuse = load_model(diffuse_name)["erg_per_ionizing_photon"]
        diagnostic = [diagnostics(mix_yields(dense, diffuse, fraction))
                      for fraction in fractions]
        axes[0].plot([row["oiii5007_oii3727"] for row in diagnostic],
                     [row["oiii_hbeta"] for row in diagnostic], lw=1.3,
                     label=fr"diffuse $\log U={logu:g}$")
    for stack, fit in zip(stacks, preferred):
        colour = colours[stack["subtype"]]
        axes[0].errorbar(stack["oiii5007_oii3727"], stack["oiii5007_hbeta"],
                         xerr=stack["oiii5007_oii3727_error"],
                         yerr=stack["oiii5007_hbeta_error"], fmt="o",
                         color=colour, ms=5, capsize=2)
        axes[0].scatter(fit["predicted_oiii5007_oii3727"],
                        fit["predicted_oiii5007_hbeta"], marker="x",
                        color=colour, s=42)
        axes[1].scatter(fit["dense_qh_fraction"],
                        fit["predicted_heii4686_hbeta"], color=colour,
                        label=stack["subtype"])
        axes[2].scatter(stack["halpha_hbeta"], stack["hbeta_hgamma"],
                        color=colour)
        axes[2].plot([stack["halpha_hbeta"],
                      fit["smooth_screen_predicted_halpha_hbeta"]],
                     [stack["hbeta_hgamma"],
                      fit["smooth_screen_predicted_hbeta_hgamma"]],
                     color=colour, lw=1)
        axes[2].scatter(fit["smooth_screen_predicted_halpha_hbeta"],
                        fit["smooth_screen_predicted_hbeta_hgamma"],
                        marker="x", color=colour, s=42)

    axes[0].set(xlabel=r"[O III] 5007 / [O II] 3727",
                ylabel=r"[O III] 5007 / H$\beta$", xlim=(0, 45), ylim=(0, 6))
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].axhline(0.1, color="black", ls="--", lw=1, label="typical upper envelope")
    axes[1].set(xlabel="dense intercepted-Q(H) fraction",
                ylabel=r"He II 4686 / H$\beta$", yscale="log")
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].set(xlabel=r"H$\alpha$/H$\beta$", ylabel=r"H$\beta$/H$\gamma$")
    axes[2].text(0.04, 0.96, "circles: stacks\n×: smooth-screen fit",
                 transform=axes[2].transAxes, va="top", fontsize=7)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Two-zone inversion of the four public LRD spectral stacks")
    fig.tight_layout()
    fig.savefig(OUT / "public_lrd_stack_two_zone_fits.pdf")
    fig.savefig(OUT / "public_lrd_stack_two_zone_fits.png", dpi=220)
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
