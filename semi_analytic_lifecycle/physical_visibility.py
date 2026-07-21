"""Physical closure connecting porous geometry to continuum and line marks."""
from __future__ import annotations

import numpy as np


H_PLANCK = 6.62607015e-27
K_BOLTZMANN = 1.380649e-16
C_LIGHT = 2.99792458e10
SIGMA_SB = 5.670374419e-5
ANGSTROM = 1.0e-8
SUBTYPE_NAMES = np.array(["bLRD", "minusLRD", "plusLRD", "xLRD"])


def blackbody_lnu_per_lbol(wavelength_angstrom, temperature_k):
    """Return L_nu/L_bol for an isotropic blackbody photosphere."""
    temperature = np.asarray(temperature_k, float)
    wavelength = float(wavelength_angstrom) * ANGSTROM
    frequency = C_LIGHT / wavelength
    exponent = H_PLANCK * frequency / (K_BOLTZMANN * temperature)
    bnu = (2.0 * H_PLANCK * frequency**3 / C_LIGHT**2
           / np.expm1(np.clip(exponent, 0.0, 700.0)))
    return np.pi * bnu / (SIGMA_SB * temperature**4)


def visibility_closure(cover_fraction, direct_uv_escape,
                       layer_gain=10.0, percolation_cover=0.59,
                       temperature_min_k=50000.0,
                       temperature_max_k=70000.0):
    """Map porous geometry to nuclear Q(H) transmission and SED hardness.

    Above the continuum-percolation covering threshold, photons encounter an
    increasing effective number of clump layers.  The only calibratable
    geometric coefficient is ``layer_gain``; the threshold is fixed.
    """
    cover, direct = np.broadcast_arrays(
        np.clip(np.asarray(cover_fraction, float), 0.0, 1.0),
        np.clip(np.asarray(direct_uv_escape, float), 0.0, 1.0),
    )
    layers = 1.0 + float(layer_gain) * np.maximum(cover - percolation_cover, 0.0)
    qh_escape = direct**layers
    # Open paths retain a somewhat harder spectrum; repeated interception
    # thermalizes the EUV toward the empirically allowed soft basis.
    temperature = temperature_min_k + (
        temperature_max_k - temperature_min_k
    ) * qh_escape
    return {
        "effective_layers": layers,
        "qh_escape_fraction": qh_escape,
        "euv_temperature_k": temperature,
    }


def two_reservoir_qh_partition(host_l5100, thermal_l5100,
                               coupling_index=2.0):
    """Partition processed Q(H) between diffuse host and dense nuclear gas.

    The observable core-to-host continuum odds provide a source-level proxy
    for the competition between nuclear and distributed ionization.  Dense
    emission requires both nuclear photon dominance and interception by the
    nuclear reprocessor, so the default closure is quadratic in those odds.
    This is one global, falsifiable coupling law rather than four subtype
    fractions.
    """
    host, thermal = np.broadcast_arrays(
        np.maximum(np.asarray(host_l5100, float), 0.0),
        np.maximum(np.asarray(thermal_l5100, float), 0.0),
    )
    if coupling_index <= 0:
        raise ValueError("coupling_index must be positive")
    dominance = np.divide(
        thermal, host, out=np.full_like(thermal, np.inf), where=host > 0
    )
    log_odds = float(coupling_index) * np.log(
        np.clip(dominance, 1.0e-12, 1.0e12)
    )
    # Stable logistic transform of the dense-to-diffuse photon odds.
    dense_fraction = np.empty_like(log_odds)
    positive = log_odds >= 0
    dense_fraction[positive] = 1.0 / (1.0 + np.exp(-log_odds[positive]))
    exp_log_odds = np.exp(log_odds[~positive])
    dense_fraction[~positive] = exp_log_odds / (1.0 + exp_log_odds)
    return {
        "core_to_host_dominance": dominance,
        "dense_to_diffuse_qh_odds": np.exp(np.clip(log_odds, -700.0, 700.0)),
        "dense_fraction_qh": dense_fraction,
        "diffuse_fraction_qh": 1.0 - dense_fraction,
    }


def continuum_marks(sfr_msunyr, lbol, lthermal, direct_agn_escape,
                    attenuation_2500_mag, redshift,
                    reprocessor_temperature_k=4050.0,
                    stellar_l5100_l2500=1.0):
    """Predict L_5100/L_2500 from stars, leaked AGN, and the cool core."""
    sfr, lbol, lthermal, direct, attenuation, redshift = np.broadcast_arrays(
        np.asarray(sfr_msunyr, float), np.asarray(lbol, float),
        np.asarray(lthermal, float), np.asarray(direct_agn_escape, float),
        np.asarray(attenuation_2500_mag, float), np.asarray(redshift, float),
    )
    uv_youth = np.clip(((1.0 + redshift) / 6.0) ** 1.5, 0.7, 2.5)
    star_2500_intrinsic = 8.0e27 * uv_youth * sfr
    star_5100_intrinsic = stellar_l5100_l2500 * star_2500_intrinsic
    agn_2500_intrinsic = 0.08 * lbol / 1.5e15 * direct
    agn_5100_intrinsic = agn_2500_intrinsic * (2500.0 / 5100.0) ** (1.0 / 3.0)
    attenuation_5100 = attenuation * (5100.0 / 2500.0) ** -1.2
    atten_2500 = 10.0 ** (-0.4 * attenuation)
    atten_5100 = 10.0 ** (-0.4 * attenuation_5100)
    thermal_2500 = lthermal * blackbody_lnu_per_lbol(
        2500.0, reprocessor_temperature_k
    )
    thermal_5100 = lthermal * blackbody_lnu_per_lbol(
        5100.0, reprocessor_temperature_k
    )
    host_2500 = star_2500_intrinsic * atten_2500
    host_5100 = star_5100_intrinsic * atten_5100
    agn_2500 = agn_2500_intrinsic * atten_2500
    agn_5100 = agn_5100_intrinsic * atten_5100
    l2500 = ((star_2500_intrinsic + agn_2500_intrinsic) * atten_2500
             + thermal_2500)
    l5100 = ((star_5100_intrinsic + agn_5100_intrinsic) * atten_5100
             + thermal_5100)
    ratio = np.divide(l5100, l2500, out=np.full_like(l5100, np.nan),
                      where=l2500 > 0)
    return {
        "l2500": l2500,
        "l5100": l5100,
        "host_l2500": host_2500,
        "host_l5100": host_5100,
        "agn_l2500": agn_2500,
        "agn_l5100": agn_5100,
        "thermal_l2500": thermal_2500,
        "thermal_l5100": thermal_5100,
        "l5100_l2500": ratio,
        "host_fraction_5100": np.divide(
            host_5100, l5100, out=np.zeros_like(l5100), where=l5100 > 0
        ),
        "thermal_fraction_5100": np.divide(
            thermal_5100, l5100, out=np.zeros_like(l5100), where=l5100 > 0
        ),
    }


def classify_lrd_subtype(l5100_l2500):
    """Apply the published 1.8, 3.1, and 6.3 continuum-ratio boundaries."""
    ratio = np.asarray(l5100_l2500, float)
    index = np.digitize(ratio, [1.8, 3.1, 6.3])
    return SUBTYPE_NAMES[index]
