#!/usr/bin/env python3
"""Semi-analytic, multi-state forward model for Little Red Dots.

The model deliberately separates four layers:

1. a cosmological halo cohort (mass growth and lognormal spin),
2. a nuclear gas/BH regulator,
3. stochastic lifecycle transitions with competing clearing risks, and
4. an observation layer producing a weighted synthetic catalogue.

It is a falsifiable population model, not a radiative-transfer calculation.
The observable mappings are transparent placeholders that can later be
replaced by calibrated emulators without changing the population machinery.
Only one global number-density factor is calibrated to the Rinaldi et al.
4.5 < z < 6.5 bin; trends and conditional distributions are predictions.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from colossus.cosmology import cosmology
from colossus.lss import bias, mass_function
from scipy.special import expit
from clump_equilibrium import step_covering
from clump_transfer import clump_transfer
from cloudy_pilot_emulator import get_pilot_emulator
from thermal_skin_emulator import (
    get_dense_thermal_correction, get_thermal_skin_emulator,
)
from physical_visibility import (
    classify_lrd_subtype, continuum_marks, two_reservoir_qh_partition,
    visibility_closure,
)
from soft_two_zone_emulator import get_soft_two_zone_emulator


OUT = Path(__file__).resolve().parent
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
cosmology.setCosmology(
    "lrd_lifecycle", {"flat": True, "H0": 70.0, "Om0": 0.3,
                      "Ob0": 0.045, "sigma8": 0.82, "ns": 0.96},
    persistence="r",
)

G = 6.67430e-8
C = 2.99792458e10
MSUN = 1.98847e33
MPC = 3.085677581e24
PC = MPC / 1.0e6
AU = 1.495978707e13
YR = 365.25 * 86400.0
MP = 1.67262192369e-24
SIGMA_T = 6.6524587321e-25
L_SUN = 3.828e33
FB = 0.156
MU_H = 1.4
ELECTRONS_PER_H = 1.16
H0_CGS = 70.0e5 / MPC
H = 0.7
OBS_Z = np.array([3.25, 5.5, 7.5, 9.5])
OBS_N = np.array([2.82e-5, 1.16e-4, 5.99e-5, 3.18e-5])
OBS_ERR = np.array([1.61e-5, 0.41e-4, 2.19e-5, 1.28e-5])

QUIESCENT, COMPACTING, EMBEDDED, CLEARING, POST = range(5)
STATE_NAMES = np.array(["quiescent", "compacting", "embedded", "clearing", "post"])
CAUSE_NAMES = np.array(["none", "gas_exhaustion", "agn_feedback", "size_growth"])
ENTRY_NONE, ENTRY_SECULAR, ENTRY_EARLY_BURST = range(3)
ENTRY_NAMES = np.array(["none", "secular_compaction", "early_burst"])


@dataclass
class Config:
    seed: int = 314159
    n_halo: int = 30000
    z_start: float = 15.0
    z_end: float = 2.0
    dt_gyr: float = 0.006
    logm_min: float = 7.0
    logm_max: float = 10.8
    spin_median: float = 0.035
    spin_sigma_ln: float = 0.50
    eps_ff: float = 0.012
    eta_sf: float = 1.5
    inflow_eff: float = 0.65
    bh_inflow_fraction: float = 0.015
    radiative_efficiency: float = 0.10
    trigger_logtau: float = -0.10
    trigger_width_dex: float = 0.25
    trigger_rate_gyr: float = 4.0
    # A physically distinct, short-lived high-z entry route.  It represents
    # stochastic gas-rich compaction (for example merger/inflow driven), not
    # an added survival tail.  Set the rate to zero to recover the single-
    # channel model exactly.
    early_burst_rate_gyr: float = 0.0
    early_burst_z_turnoff: float = 7.8
    early_burst_z_width: float = 0.45
    early_burst_growth_threshold_gyr: float = 2.5
    early_burst_growth_width_gyr: float = 0.55
    # Minimum host potential for the early channel to retain a compact,
    # rapidly supplied nucleus.  This prevents unphysical triggering in tiny
    # progenitors at z_start and makes the channel a first-threshold-crossing
    # event rather than a persistent high-redshift hazard.
    early_burst_logmh_threshold: float = 9.8
    early_burst_logmh_width: float = 0.25
    # At first entry, this fraction of the host baryons is transferred into a
    # compact nuclear reservoir.  It is then depleted explicitly; it is not a
    # luminosity multiplier or an imposed extension of the LRD lifetime.
    early_burst_reservoir_fraction: float = 0.006
    early_burst_visibility_gyr: float = 0.075
    early_burst_bh_fraction: float = 0.035
    early_burst_scatter_ln: float = 0.55
    compact_to_embedded_rate_gyr: float = 7.0
    feedback_rate_gyr: float = 1.5
    exhaustion_rate_gyr: float = 0.35
    growth_rate_gyr: float = 5.0
    clearing_rate_gyr: float = 6.0
    recurrence_rate_gyr: float = 0.10
    dust_to_gas_scale: float = 0.12
    # Optional two-scale porous reprocessor.  This layer is deliberately
    # stochastic/stationary; no unverified deterministic limit-cycle period is
    # assumed.  The host-scale reff remains separate from the red-core area.
    porous_envelope_enabled: bool = False
    envelope_radius_au: float = 1360.0
    envelope_temperature_k: float = 4050.0
    envelope_cover_scatter: float = 0.10
    channel_column_fraction: float = 1.0e-3
    channel_column_scatter_dex: float = 0.45
    # The UV leakage region need not expose the much smaller X-ray source.
    # A compact-source sightline floor encodes this geometric separation and
    # is a falsifiable assumption, not an isotropic-screen approximation.
    xray_channel_column_floor: float = 1.0e25
    envelope_dust_screen_fraction: float = 0.15
    envelope_line_escape_floor: float = 0.12
    clump_transfer_enabled: bool = False
    clump_count: int = 24
    # Opt-in co-regulated covering: the reprocessor covering is built by the
    # nuclear engine in proportion to its intrinsic continuum contrast with
    # the stellar host, instead of following the Thomson depth alone.  This
    # implements the interception law required by the eta=2 decomposition
    # (see the manuscript, Sec. 3): flux competition supplies only ~0.7 of the
    # two powers of D, and the covering odds must supply the remainder.
    # The contrast is evaluated from intrinsic 2500 A engine and stellar
    # outputs, upstream of covering itself, so the law is not circular.
    coregulated_covering_enabled: bool = False
    covering_odds_normalization: float = 3.0
    covering_odds_index: float = 1.0
    covering_odds_scatter: float = 0.35
    covering_min: float = 0.02
    # Opt-in time-resolved covering: instead of the stationary co-regulated
    # law, the covering evolves through the build/clear ODE of
    # clump_equilibrium.py, whose equilibrium is exactly C/(1-C) = x^2.
    # The relaxation timescale controls how far young or rapidly evolving
    # engines lag the equilibrium law (the hysteresis prediction).
    dynamic_covering_enabled: bool = False
    covering_relax_gyr: float = 0.08
    # Opt-in Route-B (sequential two-stage) covering law: covering requires a
    # clump to form AND survive transit, each an engine-vs-host contest with
    # per-stage odds k*x. The combined covering odds are exactly
    # (k x)^2 / (1 + 2 k x): quadratic at low-to-moderate contrast, rolling
    # to ~k x / 2 at high contrast. See the manuscript (Sec. 3).
    covering_saturating_law: bool = False
    # Opt-in Route-D (Poisson double-blocking) covering law: individual
    # clumps are marginally translucent, so opaque interception needs
    # sightlines at least two clumps deep. With areal clump density mu = k x
    # (linear in contrast from the mass budget), Poisson statistics give
    # C = 1 - exp(-mu) (1 + mu): odds ~ (k x)^2 / 2 at low contrast and
    # steeper than quadratic at high contrast (no roll-over).
    covering_poisson_law: bool = False
    # Route A-vs-C discriminator: if the second power of x comes from
    # cooling-vs-mixing fractionation (Route A), the covering odds inherit
    # the cooling function and scale with metallicity; the momentum route
    # (C) is metallicity-blind. The odds are multiplied by
    # (Z / 0.2 Zsun)^covering_metallicity_index; 0 recovers the fiducial law.
    covering_metallicity_index: float = 0.0
    # Opt-in Cloudy screening emulator. It remains disabled in every published
    # fiducial result until a full thermal grid and Q(H)/Lbol calibration are
    # available.
    cloudy_pilot_enabled: bool = False
    cloudy_emitting_logn: float = 10.0
    cloudy_emitting_log_column: float = 22.0
    cloudy_ionizing_bolometric_fraction: float = 0.5
    cloudy_mean_ionizing_energy_ryd: float = 3.26
    cloudy_sed: str = "agn_standard"
    cloudy_geometry: str = "open"
    # Optional physical two-zone emission layer. The dense fraction is the
    # fraction of incident Q(H) intercepted by compact dense clumps, not a
    # gas-mass or volume fraction. The diffuse phase uses thermal-balance
    # Cloudy skin anchors rather than the fixed-temperature pilot.
    cloudy_two_zone_enabled: bool = False
    cloudy_dense_qh_fraction: float = 0.5
    cloudy_diffuse_logu: float = -2.5
    cloudy_two_zone_dense_log_column: float = 21.0
    # Opt-in closure that predicts the dense Q(H) interception, continuum
    # subtype, and soft-EUV line ratios from the porous geometry. Unlike the
    # legacy two-zone hook, it does not assign a population-wide dense fraction.
    physical_visibility_enabled: bool = False
    visibility_layer_gain: float = 10.0
    visibility_percolation_cover: float = 0.59
    visibility_temperature_min_k: float = 50000.0
    # The public He II gate and joint [O III] stacks favor no resolved
    # subtype hardness gradient within the available soft-SED grid.
    visibility_temperature_max_k: float = 50000.0
    visibility_diffuse_logu: float = -2.1
    visibility_stellar_l5100_l2500: float = 1.0
    visibility_coupling_index: float = 2.0
    # The line-emitting/reprocessing scale is physically distinct from the
    # 20--40 pc nuclear reservoir.  When the porous envelope is active, broad
    # Balmer virial widths are evaluated here. ``virial_fwhm_factor`` absorbs
    # the line-profile geometry convention; v_circ is not a Gaussian sigma.
    broad_line_radius_au: float = 1360.0
    virial_fwhm_factor: float = 1.0
    target_density: float = 1.16e-4
    target_z_lo: float = 4.5
    target_z_hi: float = 6.5


def hz(z):
    return H0_CGS * np.sqrt(0.3 * (1.0 + z) ** 3 + 0.7)


def mdot_halo(m_msun, z):
    """Mean halo growth in Msun/yr (Fakhouri et al. 2010)."""
    return (46.1 * (m_msun / 1.0e12) ** 1.1 * (1.0 + 1.11 * z)
            * np.sqrt(0.3 * (1.0 + z) ** 3 + 0.7))


def r200c(m_msun, z):
    return (G * m_msun * MSUN / (100.0 * hz(z) ** 2)) ** (1.0 / 3.0)


def redshift_time_grid(cfg):
    t0, t1 = COSMO.age(cfg.z_start).value, COSMO.age(cfg.z_end).value
    time = np.arange(t0, t1 + cfg.dt_gyr / 2.0, cfg.dt_gyr)
    z_lookup = np.linspace(cfg.z_end, cfg.z_start + 2.0, 30000)
    age_lookup = COSMO.age(z_lookup).value
    z = np.interp(time, age_lookup[::-1], z_lookup[::-1])
    return time, z


def sample_halo_cohort(cfg, rng):
    """Importance-sample the z_start HMF uniformly in log mass."""
    logm = rng.uniform(cfg.logm_min, cfg.logm_max, cfg.n_halo)
    m_phys = 10.0 ** logm
    m_col = m_phys * H
    dndlnm = mass_function.massFunction(
        m_col, cfg.z_start, mdef="200c", model="tinker08", q_out="dndlnM"
    ) * H**3
    # Integral estimator: log10-uniform proposal density is constant.
    weight = dndlnm * np.log(10.0) * (cfg.logm_max - cfg.logm_min) / cfg.n_halo
    spin = np.exp(rng.normal(np.log(cfg.spin_median), cfg.spin_sigma_ln, cfg.n_halo))
    spin = np.clip(spin, 0.006, 0.15)
    return m_phys, weight, spin


def cohort_weights(initial_mh, current_mh, z, cfg):
    """Importance weights for the evolved cohort at a later snapshot.

    The proposal is uniform in initial log mass. Mean accretion maps it
    monotonically into current log mass, so the proposal density transforms
    with d logM_current / d logM_initial. Reweighting to the instantaneous HMF
    prevents the z_start abundance from being incorrectly carried unchanged
    to every descendant snapshot.
    """
    x0 = np.log10(initial_mh)
    x = np.log10(current_mh)
    order = np.argsort(x0)
    jacobian = np.gradient(x[order], x0[order])
    jacobian = np.clip(jacobian, 0.05, 20.0)
    jacobian_unsorted = np.empty_like(jacobian)
    jacobian_unsorted[order] = jacobian
    m_col = current_mh * H
    dndlnm = mass_function.massFunction(
        m_col, z, mdef="200c", model="tinker08", q_out="dndlnM"
    ) * H**3
    return (dndlnm * np.log(10.0) * (cfg.logm_max - cfg.logm_min)
            * jacobian_unsorted / cfg.n_halo)


def nuclear_quantities(mh, mg, radius_pc, eps_ff):
    radius = np.maximum(radius_pc, 0.3) * PC
    rho = 3.0 * np.maximum(mg, 1.0) * MSUN / (4.0 * np.pi * radius**3)
    tff_gyr = np.sqrt(3.0 * np.pi / (32.0 * G * rho)) / (1.0e9 * YR)
    tdep_gyr = tff_gyr / eps_ff
    nh = np.maximum(mg, 0.0) * MSUN / (np.pi * radius**2 * MU_H * MP)
    tau_e = SIGMA_T * ELECTRONS_PER_H * nh
    vesc = np.sqrt(2.0 * G * mh * MSUN / radius)
    return tdep_gyr, nh, tau_e, vesc


def bernoulli_hazard(rate_gyr, dt_gyr, rng):
    probability = 1.0 - np.exp(-np.maximum(rate_gyr, 0.0) * dt_gyr)
    return rng.random(np.size(probability)) < probability


def luminosity_observables(mh, mbh, mg, radius_pc, sfr, mdot_bh, z, spin,
                           state, cfg, rng, entry_channel=None,
                           cover_override=None):
    """Transparent emission/selection layer for one redshift snapshot."""
    _, nh, tau_e, vesc = nuclear_quantities(mh, mg, radius_pc, cfg.eps_ff)
    ledd = 1.26e38 * mbh
    lbol = cfg.radiative_efficiency * mdot_bh * MSUN / YR * C**2
    fedd = np.divide(lbol, ledd, out=np.zeros_like(lbol), where=ledd > 0)
    # UV from young stars plus a small escaping AGN component.
    # A modest low-metallicity/young-population UV boost at early times.
    uv_youth = np.clip(((1.0 + z) / 6.0) ** 1.5, 0.7, 2.5)
    if entry_channel is None:
        entry_channel = np.zeros(len(mh), dtype=np.int8)
    # The early channel is bright only through its dynamically evolved star
    # formation and accretion rates supplied by the finite burst reservoir.
    luv_nu_intrinsic = 8.0e27 * uv_youth * sfr + 0.08 * lbol / 1.5e15
    metallicity = np.clip(0.05 * (mh / 1e9) ** 0.25 * ((1 + z) / 7) ** -0.8,
                          0.02, 1.0)
    av = cfg.dust_to_gas_scale * metallicity * (nh / 1e22) ** 0.55
    covering = expit((np.log10(np.maximum(tau_e, 1e-8)) + 0.15) / 0.28)
    auv = np.minimum(6.0, 2.2 * av * (0.45 + 0.55 * covering))
    host_reff_pc = np.maximum(8.0, 1.7 * radius_pc)

    # Default (single-screen) observation model.  The porous branch below
    # replaces only the emission transfer, never the lifecycle dynamics.
    cover_fraction = np.zeros_like(mh)
    nh_channel = nh.copy()
    nh_xray = nh.copy()
    channel_transmission = np.ones_like(mh)
    core_reff_pc = np.zeros_like(mh)
    lthermal = np.zeros_like(mh)
    line_escape = np.ones_like(mh)
    direct_uv_escape = np.ones_like(mh)
    escaped_agn = np.ones_like(mh)
    if cfg.porous_envelope_enabled:
        active_envelope = (state == EMBEDDED) | (state == CLEARING)
        if cover_override is not None:
            # Time-resolved covering evolved in the simulate() loop through
            # the build/clear ODE; the observation layer only clips it.
            cover_fraction = np.clip(cover_override, cfg.covering_min,
                                     0.995) * active_envelope
        elif cfg.coregulated_covering_enabled:
            # Covering odds proportional to the intrinsic engine/host 2500 A
            # contrast: the reservoir feeding that powers the thermal core
            # also builds the intercepting structure.  A source whose engine
            # is continuum-subdominant cannot maintain a covering envelope.
            engine_2500 = 0.08 * lbol / 1.5e15
            star_2500 = np.maximum(8.0e27 * uv_youth * sfr, 1.0e-30)
            contrast = np.clip(engine_2500 / star_2500, 1.0e-12, 1.0e12)
            if cfg.covering_saturating_law:
                # Sequential two-stage (Route-B) odds; the exponent is the
                # combinatorial 2 and is not adjustable in this branch.
                kx = cfg.covering_odds_normalization * contrast
                ln_odds = (np.log(kx**2 / (1.0 + 2.0 * kx))
                           + rng.normal(0.0, cfg.covering_odds_scatter,
                                        len(mh)))
            elif cfg.covering_poisson_law:
                # Route-D: opaque interception requires >= 2 translucent
                # clumps along the sightline; mu = k x is the Poisson areal
                # clump density. Log-odds computed stably: ln(1 - C) =
                # -mu + ln(1 + mu) exactly.
                mu = np.clip(cfg.covering_odds_normalization * contrast,
                             1.0e-8, 60.0)
                log_one_minus_c = -mu + np.log1p(mu)
                c2 = np.clip(-np.expm1(log_one_minus_c), 1.0e-12, 1.0)
                ln_odds = (np.log(c2) - log_one_minus_c
                           + rng.normal(0.0, cfg.covering_odds_scatter,
                                        len(mh)))
            else:
                ln_odds = (np.log(cfg.covering_odds_normalization)
                           + cfg.covering_odds_index * np.log(contrast)
                           + rng.normal(0.0, cfg.covering_odds_scatter,
                                        len(mh)))
            if cfg.covering_metallicity_index != 0.0:
                ln_odds = ln_odds + cfg.covering_metallicity_index * np.log(
                    np.clip(metallicity / 0.2, 1.0e-3, 1.0e3))
            cover_fraction = np.clip(expit(ln_odds), cfg.covering_min,
                                     0.995) * active_envelope
        else:
            mean_cover = 0.58 + 0.38 * expit((np.log10(np.maximum(tau_e, 1e-8)) - 0.05) / 0.35)
            cover_fraction = np.clip(mean_cover + rng.normal(0.0, cfg.envelope_cover_scatter, len(mh)),
                                     0.35, 0.995) * active_envelope
        channel_factor = cfg.channel_column_fraction * 10**rng.normal(
            0.0, cfg.channel_column_scatter_dex, len(mh))
        nh_channel = np.maximum(1e19, nh * channel_factor)
        nh_xray = np.maximum(nh_channel, cfg.xray_channel_column_floor) * active_envelope + nh_channel * (~active_envelope)
        # Effective UV transmission through a partly ionized channel.  This is
        # a declared surrogate pending non-LTE transfer, not a photoionization
        # calculation.
        channel_transmission = np.exp(-np.minimum(30.0, 1.0e-22 * nh_channel))
        if cfg.clump_transfer_enabled:
            transfer = clump_transfer(nh, cover_fraction, rng, n_clumps=cfg.clump_count)
            channel_transmission = np.where(active_envelope, transfer["uv_transmission"], channel_transmission)
            line_escape = np.where(active_envelope, transfer["balmer_escape"], line_escape)
            nh_channel = np.where(active_envelope, transfer["mean_channel_column"], nh_channel)
            nh_xray = np.where(active_envelope, transfer["xray_effective_column"], nh_xray)
            # clump_transfer returns an area-averaged transmission, which
            # already includes the open-ray fraction.
            direct_uv_escape = np.where(
                active_envelope, channel_transmission, np.ones_like(lbol)
            )
        else:
            # Here channel_transmission is conditional on an open ray.
            direct_uv_escape = np.where(
                active_envelope,
                (1.0 - cover_fraction) * channel_transmission,
                np.ones_like(lbol),
            )
        rout = cfg.envelope_radius_au * AU
        core_reff_pc = cfg.envelope_radius_au * AU / PC * np.sqrt(cover_fraction)
        area_luminosity = 4.0 * np.pi * rout**2 * 5.670374419e-5 \
            * cfg.envelope_temperature_k**4 * cover_fraction
        if cfg.physical_visibility_enabled:
            absorbed_engine = lbol * (1.0 - direct_uv_escape)
        else:
            absorbed_engine = lbol * (cover_fraction + (1.0 - cover_fraction)
                                      * (1.0 - channel_transmission))
        # Enforce energy conservation: the cool core cannot radiate more than
        # the intercepted compact-engine luminosity.
        lthermal = np.minimum(area_luminosity, absorbed_engine) * active_envelope
        if cfg.physical_visibility_enabled:
            escaped_agn = np.clip(
                direct_uv_escape + 0.015 * cover_fraction, 0.0, 1.0
            )
        else:
            escaped_agn = ((1.0 - cover_fraction) * channel_transmission
                           + 0.015 * cover_fraction)
        # The cool reprocessor is explicitly dust-poor in this hypothesis:
        # the host dust screen acts on the stellar continuum but not as a full
        # proxy for the nuclear gas column.
        auv *= cfg.envelope_dust_screen_fraction * active_envelope + (~active_envelope)
        luv_nu_intrinsic = 8.0e27 * uv_youth * sfr + 0.08 * lbol / 1.5e15 * escaped_agn
        if not cfg.clump_transfer_enabled:
            line_escape = (cfg.envelope_line_escape_floor
                           + (1.0 - cfg.envelope_line_escape_floor)
                           * ((1.0 - cover_fraction) * channel_transmission + 0.03 * cover_fraction))
    luv_nu = np.maximum(luv_nu_intrinsic * 10 ** (-0.4 * auv), 1e15)
    muv = 51.60 - 2.5 * np.log10(luv_nu)
    vshape_color = 0.25 + 1.35 * av + 0.55 * covering + 1.25 * cover_fraction
    reff_pc = host_reff_pc

    # Geometry-predicted visibility closure. It is separate from the legacy
    # Cloudy hook so its line ratios can use the LRD-specific soft thermal grid
    # without pretending that Q(H)/Lbol is already known for that SED.
    visibility_dense_fraction_qh = np.full_like(lbol, np.nan)
    visibility_diffuse_fraction_qh = np.full_like(lbol, np.nan)
    visibility_qh_escape_fraction = np.full_like(lbol, np.nan)
    visibility_effective_layers = np.full_like(lbol, np.nan)
    visibility_euv_temperature_k = np.full_like(lbol, np.nan)
    visibility_diffuse_logu = np.full_like(lbol, np.nan)
    visibility_l5100_l2500 = np.full_like(lbol, np.nan)
    visibility_host_fraction_5100 = np.full_like(lbol, np.nan)
    visibility_thermal_fraction_5100 = np.full_like(lbol, np.nan)
    visibility_core_to_host_5100 = np.full_like(lbol, np.nan)
    visibility_m5100 = np.full_like(lbol, np.nan)
    visibility_oiii_hbeta = np.full_like(lbol, np.nan)
    visibility_oiii_oii = np.full_like(lbol, np.nan)
    visibility_heii4686_hbeta = np.full_like(lbol, np.nan)
    visibility_halpha_hbeta = np.full_like(lbol, np.nan)
    visibility_hbeta_hgamma = np.full_like(lbol, np.nan)
    visibility_subtype = np.full(len(lbol), "unclassified", dtype="<U12")
    if cfg.physical_visibility_enabled:
        if not cfg.porous_envelope_enabled:
            raise ValueError("physical_visibility_enabled requires porous_envelope_enabled")
        if cfg.visibility_layer_gain < 0:
            raise ValueError("visibility_layer_gain cannot be negative")
        if cfg.visibility_coupling_index <= 0:
            raise ValueError("visibility_coupling_index must be positive")
        if not 0.0 <= cfg.visibility_percolation_cover <= 1.0:
            raise ValueError("visibility_percolation_cover must lie in [0, 1]")
        if not (50000.0 <= cfg.visibility_temperature_min_k
                <= cfg.visibility_temperature_max_k <= 100000.0):
            raise ValueError("visibility temperature support must lie within 50--100 kK")
        closure = visibility_closure(
            cover_fraction, direct_uv_escape,
            layer_gain=cfg.visibility_layer_gain,
            percolation_cover=cfg.visibility_percolation_cover,
            temperature_min_k=cfg.visibility_temperature_min_k,
            temperature_max_k=cfg.visibility_temperature_max_k,
        )
        visibility_qh_escape_fraction = closure["qh_escape_fraction"]
        visibility_effective_layers = closure["effective_layers"]
        visibility_euv_temperature_k = closure["euv_temperature_k"]
        visibility_diffuse_logu.fill(cfg.visibility_diffuse_logu)
        continuum = continuum_marks(
            sfr, lbol, lthermal, escaped_agn, auv, z,
            reprocessor_temperature_k=cfg.envelope_temperature_k,
            stellar_l5100_l2500=cfg.visibility_stellar_l5100_l2500,
        )
        visibility_l5100_l2500 = continuum["l5100_l2500"]
        visibility_host_fraction_5100 = continuum["host_fraction_5100"]
        visibility_thermal_fraction_5100 = continuum["thermal_fraction_5100"]
        visibility_m5100 = 51.60 - 2.5 * np.log10(
            np.maximum(continuum["l5100"], 1.0)
        )
        partition = two_reservoir_qh_partition(
            continuum["host_l5100"], continuum["thermal_l5100"],
            coupling_index=cfg.visibility_coupling_index,
        )
        visibility_dense_fraction_qh = partition["dense_fraction_qh"]
        visibility_diffuse_fraction_qh = partition["diffuse_fraction_qh"]
        visibility_core_to_host_5100 = partition["core_to_host_dominance"]
        visibility_subtype = classify_lrd_subtype(visibility_l5100_l2500)
        soft_lines = get_soft_two_zone_emulator().predict(
            visibility_euv_temperature_k,
            visibility_diffuse_logu,
            visibility_dense_fraction_qh,
            clip=False,
        )
        oii = soft_lines["oii3726"] + soft_lines["oii3729"]
        visibility_oiii_hbeta = np.divide(
            soft_lines["oiii5007"], soft_lines["hbeta"],
            out=np.full_like(lbol, np.nan), where=soft_lines["hbeta"] > 0,
        )
        visibility_oiii_oii = np.divide(
            soft_lines["oiii5007"], oii,
            out=np.full_like(lbol, np.nan), where=oii > 0,
        )
        visibility_heii4686_hbeta = np.divide(
            soft_lines["heii4686"], soft_lines["hbeta"],
            out=np.full_like(lbol, np.nan), where=soft_lines["hbeta"] > 0,
        )
        visibility_halpha_hbeta = np.divide(
            soft_lines["halpha"], soft_lines["hbeta"],
            out=np.full_like(lbol, np.nan), where=soft_lines["hbeta"] > 0,
        )
        visibility_hbeta_hgamma = np.divide(
            soft_lines["hbeta"], soft_lines["hgamma"],
            out=np.full_like(lbol, np.nan), where=soft_lines["hgamma"] > 0,
        )

    # H-alpha combines SF and reprocessed AGN luminosity, then passes through
    # the same partial-covering transfer as the continuum in the porous case.
    # The historical 0.012*Lbol proxy remains the default. The opt-in branch
    # replaces only that AGN term with the Cloudy pilot yield.
    cloudy_lha_agn = np.zeros_like(lbol)
    cloudy_logu = np.full_like(lbol, np.nan)
    cloudy_balmer_decrement = np.full_like(lbol, np.nan)
    cloudy_oiii_hbeta = np.full_like(lbol, np.nan)
    cloudy_heii1640_hbeta = np.full_like(lbol, np.nan)
    cloudy_dense_fraction_qh = np.full_like(lbol, np.nan)
    agn_lha = 0.012 * lbol * covering
    if cfg.cloudy_two_zone_enabled and not cfg.cloudy_pilot_enabled:
        raise ValueError("cloudy_two_zone_enabled requires cloudy_pilot_enabled")
    if cfg.cloudy_pilot_enabled:
        if not (0.0 <= cfg.cloudy_ionizing_bolometric_fraction <= 1.0):
            raise ValueError("cloudy_ionizing_bolometric_fraction must be in [0, 1]")
        rydberg_erg = 2.1798723611035e-11
        qh = (cfg.cloudy_ionizing_bolometric_fraction * lbol /
              (cfg.cloudy_mean_ionizing_energy_ryd * rydberg_erg))
        if cfg.porous_envelope_enabled:
            emission_radius = np.full_like(lbol, cfg.envelope_radius_au * AU)
        else:
            emission_radius = np.maximum(radius_pc, 0.3) * PC
        ionization_parameter = qh / (
            4.0 * np.pi * emission_radius**2 * 10.0**cfg.cloudy_emitting_logn * C
        )
        cloudy_logu = np.log10(np.maximum(ionization_parameter, 1.0e-99))
        emulator = get_pilot_emulator()
        dense_log_column = (cfg.cloudy_two_zone_dense_log_column
                            if cfg.cloudy_two_zone_enabled
                            else cfg.cloudy_emitting_log_column)
        dense_lines = emulator.predict(
            logn=np.full_like(lbol, cfg.cloudy_emitting_logn),
            logu=cloudy_logu,
            lognh=np.full_like(lbol, dense_log_column),
            metallicity=metallicity,
            sed=cfg.cloudy_sed,
            geometry=cfg.cloudy_geometry,
            clip=True,
        )
        cloudy_lines = dense_lines
        if cfg.cloudy_two_zone_enabled:
            if not 0.0 <= cfg.cloudy_dense_qh_fraction <= 1.0:
                raise ValueError("cloudy_dense_qh_fraction must be in [0, 1]")
            thermal = get_thermal_skin_emulator()
            dense_correction = get_dense_thermal_correction()
            dense_lines = {
                line: values * dense_correction[line]
                for line, values in dense_lines.items()
            }
            diffuse_lines = thermal.predict(
                logu=np.full_like(lbol, cfg.cloudy_diffuse_logu),
                metallicity=metallicity,
                clip=True,
            )
            dense_fraction = cfg.cloudy_dense_qh_fraction
            cloudy_lines = {
                line: dense_fraction * dense_lines[line]
                + (1.0 - dense_fraction) * diffuse_lines[line]
                for line in dense_lines
            }
            cloudy_dense_fraction_qh.fill(dense_fraction)
        cloudy_lha_agn = qh * cloudy_lines["halpha"] * covering
        agn_lha = cloudy_lha_agn
        cloudy_balmer_decrement = np.divide(
            cloudy_lines["halpha"], cloudy_lines["hbeta"],
            out=np.full_like(lbol, np.nan), where=cloudy_lines["hbeta"] > 0,
        )
        cloudy_oiii_hbeta = np.divide(
            cloudy_lines["oiii5007"], cloudy_lines["hbeta"],
            out=np.full_like(lbol, np.nan), where=cloudy_lines["hbeta"] > 0,
        )
        cloudy_heii1640_hbeta = np.divide(
            cloudy_lines["heii1640"], cloudy_lines["hbeta"],
            out=np.full_like(lbol, np.nan), where=cloudy_lines["hbeta"] > 0,
        )
    lha = (1.27e41 * sfr + agn_lha) * line_escape
    electron_width = 1100.0 * np.sqrt(np.clip(tau_e, 0, 12))
    reservoir_line_radius = np.maximum(radius_pc, 0.3) * PC
    if cfg.porous_envelope_enabled:
        broad_line_radius = np.where(
            active_envelope,
            cfg.broad_line_radius_au * AU,
            reservoir_line_radius,
        )
    else:
        broad_line_radius = reservoir_line_radius
    virial_fwhm = (cfg.virial_fwhm_factor
                   * np.sqrt(G * mbh * MSUN / broad_line_radius) / 1e5)
    fwhm = np.sqrt(virial_fwhm**2 + electron_width**2)
    line_kurtosis = 3.0 + 2.2 * tau_e / (1.0 + tau_e)

    # X-ray weakness combines absorption and super-Eddington coronal suppression.
    lx_intrinsic = 0.025 * lbol / (1.0 + np.maximum(fedd - 1.0, 0.0))
    transmission = np.exp(-np.minimum(30.0, nh * 2.0e-24))
    if cfg.porous_envelope_enabled:
        # The compact X-ray corona is behind the dense core sightline even
        # when extended UV-emitting/scattering regions leak through channels.
        transmission = np.exp(-np.minimum(30.0, nh_xray * 2.0e-24))
    lx = lx_intrinsic * transmission
    variability_rms = (0.16 / np.sqrt(1.0 + np.maximum(fedd, 0.0))
                       * np.exp(-0.35 * tau_e))
    if cfg.porous_envelope_enabled:
        variability_rms *= (0.10 + 0.90 * (1.0 - cover_fraction) * channel_transmission)
    # A simple radio prediction tied to SF; no jet component in the fiducial model.
    lradio_14 = 1.6e28 * sfr

    compact_probability = expit((300.0 - reff_pc) / 45.0)
    color_probability = expit((vshape_color - 1.0) / 0.08)
    magnitude_probability = expit((-18.5 - muv) / 0.25)
    stage_probability = np.where(state == EMBEDDED, 1.0,
                                 np.where(state == CLEARING, 0.55, 0.0))
    depth_probability = expit((27.5 - (muv + 47.0 - 5*np.log10(1 + z))) / 0.7)
    p_select = (stage_probability * compact_probability * color_probability
                * magnitude_probability * depth_probability)
    selected = rng.random(len(mh)) < p_select
    return {
        "nh": nh, "tau_e": tau_e, "metallicity": metallicity, "av": av, "muv": muv,
        "color": vshape_color, "reff_pc": reff_pc, "lha": lha,
        "fwhm": fwhm, "line_kurtosis": line_kurtosis, "lbol": lbol,
        "fedd": fedd, "lx": lx, "variability_rms": variability_rms,
        "lradio_14": lradio_14, "p_select": p_select, "selected": selected,
        "vesc_kms": vesc / 1e5, "cover_fraction": cover_fraction,
        "nh_channel": nh_channel, "channel_transmission": channel_transmission,
        "direct_uv_escape": direct_uv_escape,
        "nh_xray": nh_xray,
        "broad_line_radius_au": broad_line_radius / AU,
        "core_reff_pc": core_reff_pc, "lthermal": lthermal,
        "cloudy_lha_agn": cloudy_lha_agn, "cloudy_logu": cloudy_logu,
        "cloudy_balmer_decrement": cloudy_balmer_decrement,
        "cloudy_oiii_hbeta": cloudy_oiii_hbeta,
        "cloudy_heii1640_hbeta": cloudy_heii1640_hbeta,
        "cloudy_dense_fraction_qh": cloudy_dense_fraction_qh,
        "visibility_dense_fraction_qh": visibility_dense_fraction_qh,
        "visibility_diffuse_fraction_qh": visibility_diffuse_fraction_qh,
        "visibility_qh_escape_fraction": visibility_qh_escape_fraction,
        "visibility_effective_layers": visibility_effective_layers,
        "visibility_euv_temperature_k": visibility_euv_temperature_k,
        "visibility_diffuse_logu": visibility_diffuse_logu,
        "visibility_l5100_l2500": visibility_l5100_l2500,
        "visibility_host_fraction_5100": visibility_host_fraction_5100,
        "visibility_thermal_fraction_5100": visibility_thermal_fraction_5100,
        "visibility_core_to_host_5100": visibility_core_to_host_5100,
        "visibility_m5100": visibility_m5100,
        "visibility_oiii_hbeta": visibility_oiii_hbeta,
        "visibility_oiii_oii": visibility_oiii_oii,
        "visibility_heii4686_hbeta": visibility_heii4686_hbeta,
        "visibility_halpha_hbeta": visibility_halpha_hbeta,
        "visibility_hbeta_hgamma": visibility_hbeta_hgamma,
        "visibility_subtype": visibility_subtype,
    }


def weighted_quantile(values, weights, q=0.5):
    if len(values) == 0 or np.sum(weights) <= 0:
        return np.nan
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return float(np.interp(q, cdf, v))


def weighted_corr(x, y, w):
    if len(x) < 3 or np.sum(w) <= 0:
        return np.nan
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    return float(cov / np.sqrt(max(vx * vy, 1e-30)))


def simulate(cfg: Config, store_catalog=True):
    rng = np.random.default_rng(cfg.seed)
    time, redshift = redshift_time_grid(cfg)
    mh, weight, spin = sample_halo_cohort(cfg, rng)
    initial_mh = mh.copy()
    mbh = np.maximum(1e3, 1.5e-5 * mh * np.exp(rng.normal(0, 0.7, cfg.n_halo)))
    mg = np.maximum(1e3, 2e-5 * FB * mh)
    burst_reservoir = np.zeros(cfg.n_halo)
    state = np.full(cfg.n_halo, QUIESCENT, dtype=np.int8)
    state_age = np.zeros(cfg.n_halo)
    exit_cause = np.zeros(cfg.n_halo, dtype=np.int8)
    # Persist the route into the active episode; it becomes a predicted mark
    # in the synthetic catalogue, rather than an unobservable fit component.
    entry_channel = np.zeros(cfg.n_halo, dtype=np.int8)
    if cfg.early_burst_rate_gyr > 0:
        burst_propensity = np.exp(rng.normal(-0.5 * cfg.early_burst_scatter_ln**2,
                                             cfg.early_burst_scatter_ln, cfg.n_halo))
    else:
        # Keep the single-channel model bitwise reproducible when the new
        # channel is disabled.
        burst_propensity = np.ones(cfg.n_halo)
    radius_pc = spin**2 * r200c(mh, cfg.z_start) / PC
    if cfg.dynamic_covering_enabled:
        if not cfg.porous_envelope_enabled:
            raise ValueError("dynamic_covering_enabled requires porous_envelope_enabled")
        # Persistent covering state, evolved through the build/clear ODE.
        # The per-halo contrast propensity is frozen (drawn once) so the
        # scatter is a population property, not per-snapshot noise.
        cover_state = np.full(cfg.n_halo, cfg.covering_min)
        covering_propensity = np.exp(
            rng.normal(0.0, cfg.covering_odds_scatter, cfg.n_halo))
    else:
        cover_state = None
        covering_propensity = None
    snapshot_z = np.array([9.5, 7.5, 6.0, 5.5, 5.0, 4.0, 3.25, 2.5])
    snapshot_indices = {int(np.argmin(abs(redshift - z))): z for z in snapshot_z}
    summaries, catalog = [], []
    transitions = np.zeros((5, 5), dtype=int)
    cause_counts = np.zeros(4, dtype=int)
    entry_counts = np.zeros(3, dtype=int)

    for it, (t, z) in enumerate(zip(time, redshift)):
        dt = cfg.dt_gyr
        dmh_dt = mdot_halo(mh, z) * 1e9
        mh += dmh_dt * dt
        halo_inflow = cfg.inflow_eff * FB * dmh_dt

        base_radius = spin**2 * r200c(mh, z) / PC
        radius_factor = np.choose(state, [1.0, 0.48, 0.22, 0.42, 0.90])
        target_radius = np.maximum(2.0, base_radius * radius_factor)
        relax_gyr = np.choose(state, [0.25, 0.08, 0.06, 0.08, 0.25])
        radius_pc += (target_radius - radius_pc) * np.minimum(dt / relax_gyr, 1.0)

        tdep, nh, tau_e, vesc = nuclear_quantities(mh, mg, radius_pc, cfg.eps_ff)
        nuclear_sfr = mg / np.maximum(tdep, 1e-5)
        burst_sfr = burst_reservoir / max(cfg.early_burst_visibility_gyr, 1e-5)
        sfr = nuclear_sfr + burst_sfr
        active = (state == COMPACTING) | (state == EMBEDDED) | (state == CLEARING)
        inflow_fraction = np.choose(state, [0.03, 0.16, 0.24, 0.09, 0.04])
        nuclear_inflow = inflow_fraction * halo_inflow
        mdot_edd = 2.2e-8 * mbh / max(cfg.radiative_efficiency / 0.1, 1e-5) * 1e9
        mdot_bh_nuclear = (cfg.bh_inflow_fraction * nuclear_inflow
                           + active * 0.002 * mg / np.maximum(tdep, 1e-4))
        mdot_bh_burst_requested = cfg.early_burst_bh_fraction * burst_sfr
        mdot_bh = np.minimum(mdot_bh_nuclear + mdot_bh_burst_requested,
                             4.0 * mdot_edd)
        # Attribute a capped accretion flow to the burst only after the
        # ordinary nuclear supply is accounted for.
        mdot_bh_burst = np.minimum(mdot_bh_burst_requested, mdot_bh)
        mdot_bh_nuclear = mdot_bh - mdot_bh_burst
        lbol = cfg.radiative_efficiency * (mdot_bh / 1e9) * MSUN / YR * C**2
        binding_rate = (mg * MSUN * vesc**2 / 2.0) / (0.03 * 1e9 * YR)
        feedback_loading = active * np.clip(0.005 * lbol / np.maximum(binding_rate, 1e20), 0, 8)
        gas_loss = ((1.0 + cfg.eta_sf) * nuclear_sfr + mdot_bh_nuclear
                    + feedback_loading * nuclear_sfr)
        burst_loss = ((1.0 + cfg.eta_sf) * burst_sfr + mdot_bh_burst
                      + feedback_loading * burst_sfr)
        mg += (nuclear_inflow - gas_loss) * dt
        mg = np.clip(mg, 1e2, 0.12 * FB * mh)
        burst_reservoir = np.maximum(0.0, burst_reservoir - burst_loss * dt)
        mbh += (1.0 - cfg.radiative_efficiency) * mdot_bh * dt

        if cfg.dynamic_covering_enabled:
            # Evolve the covering through the build/clear ODE wherever an
            # envelope can exist; the equilibrium of this update is exactly
            # the stationary C/(1-C) = x^2 law.
            envelope_states = ((state == COMPACTING) | (state == EMBEDDED)
                               | (state == CLEARING))
            uv_youth_step = np.clip(((1.0 + z) / 6.0) ** 1.5, 0.7, 2.5)
            engine_2500 = 0.08 * lbol / 1.5e15
            star_2500 = np.maximum(8.0e27 * uv_youth_step * sfr / 1e9, 1.0e-30)
            contrast = np.clip(covering_propensity * engine_2500 / star_2500,
                               1.0e-12, 1.0e12)
            cover_state = np.where(
                envelope_states,
                step_covering(cover_state, contrast, cfg.covering_relax_gyr, dt),
                cover_state,
            )

        old_state = state.copy()
        state_age += dt
        specific_growth = dmh_dt / np.maximum(mh, 1.0)
        low_spin = expit((np.log10(0.045) - np.log10(spin)) / 0.18)
        tau_trigger = expit((np.log10(np.maximum(tau_e, 1e-8))
                             - cfg.trigger_logtau) / cfg.trigger_width_dex)
        growth_boost = np.clip(specific_growth / 1.5, 0.2, 3.0)

        qmask = old_state == QUIESCENT
        secular_rate = cfg.trigger_rate_gyr * low_spin * tau_trigger * growth_boost
        # This channel is restricted to rapidly assembling systems at early
        # epochs.  The logistic gate is deliberately smooth: its location and
        # width are parameters to be tested, not a hard redshift cut.
        early_gate = expit((z - cfg.early_burst_z_turnoff) / cfg.early_burst_z_width)
        burst_growth = expit((specific_growth - cfg.early_burst_growth_threshold_gyr)
                             / cfg.early_burst_growth_width_gyr)
        burst_mass = expit((np.log10(mh) - cfg.early_burst_logmh_threshold)
                            / cfg.early_burst_logmh_width)
        burst_rate = (cfg.early_burst_rate_gyr * early_gate * burst_growth
                      * burst_mass * tau_trigger * burst_propensity)
        total_entry_rate = secular_rate + burst_rate
        q_to_c = qmask & bernoulli_hazard(total_entry_rate, dt, rng)
        state[q_to_c] = COMPACTING
        if cfg.dynamic_covering_enabled and np.any(q_to_c):
            # A fresh episode starts with an unbuilt envelope.
            cover_state[q_to_c] = cfg.covering_min
        if np.any(q_to_c):
            p_burst = burst_rate[q_to_c] / np.maximum(total_entry_rate[q_to_c], 1e-30)
            is_burst = rng.random(np.sum(q_to_c)) < p_burst
            channels = np.where(is_burst, ENTRY_EARLY_BURST, ENTRY_SECULAR)
            entry_channel[q_to_c] = channels
            np.add.at(entry_counts, channels, 1)
            burst_entries = q_to_c.copy()
            burst_entries[q_to_c] = is_burst
            # A burst funnels a finite share of the halo gas into the nucleus.
            # The combined nuclear reservoirs never exceed the adopted cap.
            available = np.maximum(0.0, 0.12 * FB * mh[burst_entries]
                                   - mg[burst_entries] - burst_reservoir[burst_entries])
            deposited = np.minimum(cfg.early_burst_reservoir_fraction * FB
                                   * mh[burst_entries], available)
            burst_reservoir[burst_entries] += deposited

        cmask = old_state == COMPACTING
        c_rate = cfg.compact_to_embedded_rate_gyr * tau_trigger
        c_to_e = cmask & (state_age > 0.025) & bernoulli_hazard(c_rate, dt, rng)
        state[c_to_e] = EMBEDDED

        emask = old_state == EMBEDDED
        gas_fraction = mg / np.maximum(FB * mh, 1.0)
        ledd = 1.26e38 * mbh
        fedd = np.divide(lbol, ledd, out=np.zeros_like(lbol), where=ledd > 0)
        rates = np.vstack([
            cfg.exhaustion_rate_gyr * expit((0.002 - gas_fraction) / 0.0005),
            cfg.feedback_rate_gyr * expit((fedd - 0.8) / 0.35),
            cfg.growth_rate_gyr * expit((np.log10(mh) - 10.9) / 0.22),
        ]).T
        total_rate = rates.sum(axis=1)
        exits = emask & (state_age > 0.04) & bernoulli_hazard(total_rate, dt, rng)
        if np.any(exits):
            choice_draw = rng.random(np.sum(exits)) * total_rate[exits]
            cumulative = np.cumsum(rates[exits], axis=1)
            causes = 1 + np.sum(choice_draw[:, None] > cumulative, axis=1)
            exit_cause[exits] = causes
            np.add.at(cause_counts, causes, 1)
            state[exits] = CLEARING

        clear_mask = old_state == CLEARING
        clear_to_post = clear_mask & (state_age > 0.03) & bernoulli_hazard(
            np.full(cfg.n_halo, cfg.clearing_rate_gyr), dt, rng)
        state[clear_to_post] = POST

        post_mask = old_state == POST
        recurrence = (post_mask & (state_age > 0.20)
                      & bernoulli_hazard(cfg.recurrence_rate_gyr * low_spin
                                         * growth_boost * tau_trigger, dt, rng))
        state[recurrence] = COMPACTING
        exit_cause[recurrence] = 0
        entry_channel[recurrence] = ENTRY_SECULAR
        entry_counts[ENTRY_SECULAR] += int(np.sum(recurrence))

        changed = state != old_state
        if np.any(changed):
            np.add.at(transitions, (old_state[changed], state[changed]), 1)
            state_age[changed] = 0.0

        if it in snapshot_indices:
            snapshot_weight = cohort_weights(initial_mh, mh, z, cfg)
            obs = luminosity_observables(mh, mbh, mg, radius_pc, sfr / 1e9,
                                         mdot_bh / 1e9, z, spin, state, cfg, rng,
                                         entry_channel=entry_channel,
                                         cover_override=cover_state)
            # Integrate over selection probability for stable population
            # predictions. A Bernoulli realization is still returned by the
            # observation layer for mock-survey applications.
            sel = obs["p_select"] > 1e-6
            w = snapshot_weight[sel] * obs["p_select"][sel]
            m_col = mh[sel] * H
            hbias = (bias.haloBias(m_col, z, mdef="200c", model="tinker10")
                     if np.any(sel) else np.array([]))
            raw_density = float(np.sum(w))
            row = {
                "z": float(z), "raw_density": raw_density,
                "selection_support_count": int(np.sum(sel)),
                "realized_selected_count": int(np.sum(obs["selected"])),
                "selection_effective_sample_size": float(
                    np.sum(w)**2 / np.sum(w**2)) if np.sum(w**2) > 0 else 0.0,
                "embedded_fraction": float(np.average(state == EMBEDDED,
                                                        weights=snapshot_weight)),
                "selected_embedded_fraction": float(np.average(
                    state[sel] == EMBEDDED, weights=w)) if len(w) else np.nan,
                "selected_early_burst_fraction": float(np.average(
                    entry_channel[sel] == ENTRY_EARLY_BURST, weights=w)) if len(w) else np.nan,
                "selected_burst_visible_fraction": float(np.average(
                    burst_reservoir[sel] > 1e4, weights=w)) if len(w) else np.nan,
                "median_logmh": weighted_quantile(np.log10(mh[sel]), w),
                "median_logmbh": weighted_quantile(np.log10(mbh[sel]), w),
                "median_logmgas": weighted_quantile(np.log10(mg[sel]), w),
                "median_reff_pc": weighted_quantile(obs["reff_pc"][sel], w),
                "median_core_reff_au": weighted_quantile(obs["core_reff_pc"][sel] * PC / AU, w),
                "median_cover_fraction": weighted_quantile(obs["cover_fraction"][sel], w),
                "median_log_nh_channel": weighted_quantile(np.log10(obs["nh_channel"][sel]), w),
                "median_log_nh_xray": weighted_quantile(np.log10(obs["nh_xray"][sel]), w),
                "median_muv": weighted_quantile(obs["muv"][sel], w),
                "median_log_nh": weighted_quantile(np.log10(obs["nh"][sel]), w),
                "median_tau_e": weighted_quantile(obs["tau_e"][sel], w),
                "median_fwhm_kms": weighted_quantile(obs["fwhm"][sel], w),
                "median_broad_line_radius_au": weighted_quantile(
                    obs["broad_line_radius_au"][sel], w),
                "median_log_lx": weighted_quantile(np.log10(np.maximum(obs["lx"][sel], 1)), w),
                "median_variability_rms": weighted_quantile(obs["variability_rms"][sel], w),
                "median_cloudy_logu": weighted_quantile(obs["cloudy_logu"][sel], w),
                "median_cloudy_balmer_decrement": weighted_quantile(
                    obs["cloudy_balmer_decrement"][sel], w),
                "median_cloudy_oiii_hbeta": weighted_quantile(
                    obs["cloudy_oiii_hbeta"][sel], w),
                "median_cloudy_heii1640_hbeta": weighted_quantile(
                    obs["cloudy_heii1640_hbeta"][sel], w),
                "median_cloudy_dense_fraction_qh": weighted_quantile(
                    obs["cloudy_dense_fraction_qh"][sel], w),
                "median_visibility_dense_fraction_qh": weighted_quantile(
                    obs["visibility_dense_fraction_qh"][sel], w),
                "median_visibility_diffuse_fraction_qh": weighted_quantile(
                    obs["visibility_diffuse_fraction_qh"][sel], w),
                "median_visibility_qh_escape_fraction": weighted_quantile(
                    obs["visibility_qh_escape_fraction"][sel], w),
                "median_visibility_euv_temperature_k": weighted_quantile(
                    obs["visibility_euv_temperature_k"][sel], w),
                "median_visibility_l5100_l2500": weighted_quantile(
                    obs["visibility_l5100_l2500"][sel], w),
                "median_visibility_core_to_host_5100": weighted_quantile(
                    obs["visibility_core_to_host_5100"][sel], w),
                "median_visibility_m5100": weighted_quantile(
                    obs["visibility_m5100"][sel], w),
                "median_visibility_oiii_hbeta": weighted_quantile(
                    obs["visibility_oiii_hbeta"][sel], w),
                "median_visibility_oiii_oii": weighted_quantile(
                    obs["visibility_oiii_oii"][sel], w),
                "median_visibility_heii4686_hbeta": weighted_quantile(
                    obs["visibility_heii4686_hbeta"][sel], w),
                "visibility_subtype_fraction_xLRD": float(np.average(
                    obs["visibility_subtype"][sel] == "xLRD", weights=w
                )) if len(w) and cfg.physical_visibility_enabled else np.nan,
                "visibility_subtype_fraction_plusLRD": float(np.average(
                    obs["visibility_subtype"][sel] == "plusLRD", weights=w
                )) if len(w) and cfg.physical_visibility_enabled else np.nan,
                "visibility_subtype_fraction_minusLRD": float(np.average(
                    obs["visibility_subtype"][sel] == "minusLRD", weights=w
                )) if len(w) and cfg.physical_visibility_enabled else np.nan,
                "visibility_subtype_fraction_bLRD": float(np.average(
                    obs["visibility_subtype"][sel] == "bLRD", weights=w
                )) if len(w) and cfg.physical_visibility_enabled else np.nan,
                "mean_halo_bias": float(np.average(hbias, weights=w)) if len(w) else np.nan,
                "corr_nh_xweak": weighted_corr(
                    np.log10(obs["nh"][sel]),
                    np.log10(np.maximum(obs["lx"][sel] / np.maximum(obs["lha"][sel], 1), 1e-20)), w),
                "corr_tau_variability": weighted_corr(obs["tau_e"][sel],
                                                        obs["variability_rms"][sel], w),
                "corr_size_fwhm": weighted_corr(np.log10(obs["reff_pc"][sel]),
                                                  np.log10(obs["fwhm"][sel]), w),
            }
            # Intrinsic (selection-free) densities of the nuclear-active,
            # UV-bright population, computed on the FULL array before the
            # p_select > 1e-6 storage floor: the catalogue cannot recover
            # these (Letter Sec. 5 headroom), so they are emitted here.
            active_states = (state == EMBEDDED) | (state == CLEARING)
            bright = active_states & (obs["muv"] < -18.5)
            row["density_active_muv185_intrinsic"] = float(
                np.sum(snapshot_weight[bright]))
            row["density_active_muv185_selected"] = float(
                np.sum(snapshot_weight[bright] * obs["p_select"][bright]))
            if cfg.physical_visibility_enabled:
                for st in ("xLRD", "plusLRD", "minusLRD", "bLRD"):
                    ms = bright & (obs["visibility_subtype"] == st)
                    row[f"density_muv185_intrinsic_{st}"] = float(
                        np.sum(snapshot_weight[ms]))
                    row[f"density_muv185_selected_{st}"] = float(
                        np.sum(snapshot_weight[ms] * obs["p_select"][ms]))
            summaries.append(row)
            if store_catalog and np.any(sel):
                selected_indices = np.flatnonzero(sel)
                # Store at most 2000 per snapshot while preserving weighted tails.
                if len(selected_indices) > 2000:
                    selected_indices = rng.choice(selected_indices, 2000, replace=False)
                for j in selected_indices:
                    catalog.append({
                        "z": float(z),
                        "weight_raw": float(snapshot_weight[j] * obs["p_select"][j]),
                        "state": str(STATE_NAMES[state[j]]),
                        "state_age_gyr": float(state_age[j]),
                        "exit_cause": str(CAUSE_NAMES[exit_cause[j]]),
                        "entry_channel": str(ENTRY_NAMES[entry_channel[j]]),
                        "logmh": float(np.log10(mh[j])), "spin": float(spin[j]),
                        "logmbh": float(np.log10(mbh[j])),
                        "logmgas": float(np.log10(mg[j])),
                        "logmburst": float(np.log10(max(burst_reservoir[j], 1.0))),
                        "sfr_msunyr": float(sfr[j] / 1e9),
                        "log_lbol": float(np.log10(max(obs["lbol"][j], 1.0))),
                        "reff_pc": float(obs["reff_pc"][j]),
                        "core_reff_au": float(obs["core_reff_pc"][j] * PC / AU),
                        "cover_fraction": float(obs["cover_fraction"][j]),
                        "lognh_channel": float(np.log10(obs["nh_channel"][j])),
                        "lognh_xray": float(np.log10(obs["nh_xray"][j])),
                        "channel_transmission": float(obs["channel_transmission"][j]),
                        "direct_uv_escape": float(obs["direct_uv_escape"][j]),
                        "log_lthermal": float(np.log10(max(obs["lthermal"][j], 1.0))),
                        "muv": float(obs["muv"][j]),
                        "color": float(obs["color"][j]), "lognh": float(np.log10(obs["nh"][j])),
                        "metallicity_zsun": float(obs["metallicity"][j]),
                        "tau_e": float(obs["tau_e"][j]), "fwhm_kms": float(obs["fwhm"][j]),
                        "broad_line_radius_au": float(obs["broad_line_radius_au"][j]),
                        "line_kurtosis": float(obs["line_kurtosis"][j]),
                        "log_lha": float(np.log10(max(obs["lha"][j], 1))),
                        "log_cloudy_lha_agn": float(np.log10(max(obs["cloudy_lha_agn"][j], 1))),
                        "cloudy_logu": float(obs["cloudy_logu"][j]),
                        "cloudy_balmer_decrement": float(obs["cloudy_balmer_decrement"][j]),
                        "cloudy_oiii_hbeta": float(obs["cloudy_oiii_hbeta"][j]),
                        "cloudy_heii1640_hbeta": float(obs["cloudy_heii1640_hbeta"][j]),
                        "cloudy_dense_fraction_qh": float(
                            obs["cloudy_dense_fraction_qh"][j]),
                        "visibility_dense_fraction_qh": float(
                            obs["visibility_dense_fraction_qh"][j]),
                        "visibility_diffuse_fraction_qh": float(
                            obs["visibility_diffuse_fraction_qh"][j]),
                        "visibility_qh_escape_fraction": float(
                            obs["visibility_qh_escape_fraction"][j]),
                        "visibility_effective_layers": float(
                            obs["visibility_effective_layers"][j]),
                        "visibility_euv_temperature_k": float(
                            obs["visibility_euv_temperature_k"][j]),
                        "visibility_diffuse_logu": float(
                            obs["visibility_diffuse_logu"][j]),
                        "visibility_l5100_l2500": float(
                            obs["visibility_l5100_l2500"][j]),
                        "visibility_host_fraction_5100": float(
                            obs["visibility_host_fraction_5100"][j]),
                        "visibility_thermal_fraction_5100": float(
                            obs["visibility_thermal_fraction_5100"][j]),
                        "visibility_core_to_host_5100": float(
                            obs["visibility_core_to_host_5100"][j]),
                        "visibility_m5100": float(obs["visibility_m5100"][j]),
                        "visibility_oiii_hbeta": float(
                            obs["visibility_oiii_hbeta"][j]),
                        "visibility_oiii_oii": float(obs["visibility_oiii_oii"][j]),
                        "visibility_heii4686_hbeta": float(
                            obs["visibility_heii4686_hbeta"][j]),
                        "visibility_halpha_hbeta": float(
                            obs["visibility_halpha_hbeta"][j]),
                        "visibility_hbeta_hgamma": float(
                            obs["visibility_hbeta_hgamma"][j]),
                        "visibility_subtype": str(obs["visibility_subtype"][j]),
                        "log_lx": float(np.log10(max(obs["lx"][j], 1))),
                        "fedd": float(obs["fedd"][j]),
                        "variability_rms": float(obs["variability_rms"][j]),
                        "p_select": float(obs["p_select"][j]),
                    })

    # One scalar calibration. It absorbs cohort incompleteness and unknown survey area.
    calibration_rows = [r for r in summaries if cfg.target_z_lo <= r["z"] <= cfg.target_z_hi]
    reference = np.mean([r["raw_density"] for r in calibration_rows])
    normalization = cfg.target_density / reference if reference > 0 else np.nan
    for row in summaries:
        row["predicted_density"] = row["raw_density"] * normalization
    for row in catalog:
        row["weight_cMpc3"] = row.pop("weight_raw") * normalization
    predicted_at_data = np.array([
        min(summaries, key=lambda row: abs(row["z"] - z))["predicted_density"]
        for z in OBS_Z
    ])
    pulls = (predicted_at_data - OBS_N) / OBS_ERR
    demographic_fit = {
        "observed_z": OBS_Z.tolist(), "observed_density": OBS_N.tolist(),
        "observed_error": OBS_ERR.tolist(), "predicted_density": predicted_at_data.tolist(),
        "pulls_sigma": pulls.tolist(), "chi2": float(np.sum(pulls**2)),
        "nominal_dof_after_one_normalization": 3,
    }
    return {
        "config": asdict(cfg), "normalization": float(normalization),
        "summaries": summaries, "catalog": catalog,
        "transition_counts": transitions.tolist(),
        "exit_cause_counts": {
            str(CAUSE_NAMES[i]): int(cause_counts[i]) for i in range(1, 4)
        },
        "entry_channel_counts": {
            str(ENTRY_NAMES[i]): int(entry_counts[i]) for i in range(1, 3)
        },
        "demographic_fit": demographic_fit,
    }


def save_catalog_csv(rows, path):
    import csv
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(result, suffix="fiducial"):
    rows = sorted(result["summaries"], key=lambda r: r["z"])
    z = np.array([r["z"] for r in rows])
    fig, ax = plt.subplots(2, 3, figsize=(10.2, 6.3), sharex=True)
    ax[0, 0].plot(z, [r["predicted_density"] for r in rows], "o-", color="#2864a5")
    ax[0, 0].errorbar(OBS_Z, OBS_N, yerr=OBS_ERR, fmt="s", ms=4,
                      color="black", capsize=2, label="observed bins")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_ylabel(r"$n_{\rm sel}$ [cMpc$^{-3}$]")
    ax[0, 0].legend(frameon=False, fontsize=7)
    ax[0, 1].plot(z, [r["median_logmh"] for r in rows], "o-", color="#e47c22")
    ax[0, 1].set_ylabel(r"median $\log M_h/M_\odot$")
    ax[0, 2].plot(z, [r["mean_halo_bias"] for r in rows], "o-", color="#3a913f")
    ax[0, 2].set_ylabel("mean halo bias")
    ax[1, 0].plot(z, [r["median_log_nh"] for r in rows], "o-", color="#7c4aa5")
    ax[1, 0].axhline(np.log10(1 / SIGMA_T), color="0.4", ls=":")
    ax[1, 0].set_ylabel(r"median $\log N_H$ [cm$^{-2}$]")
    ax[1, 1].plot(z, [r["median_reff_pc"] for r in rows], "o-", color="#b44e57")
    ax[1, 1].set_ylabel(r"median $R_{\rm eff}$ [pc]")
    ax[1, 2].plot(z, [r["median_variability_rms"] for r in rows], "o-", color="#2b8c8c")
    ax[1, 2].set_ylabel("median fractional variability")
    for a in ax.ravel():
        a.set_xlabel("redshift z")
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Semi-analytic LRD lifecycle: joint forward predictions", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / f"lifecycle_predictions_{suffix}.pdf")
    fig.savefig(OUT / f"lifecycle_predictions_{suffix}.png", dpi=220)
    plt.close(fig)

    cat = result["catalog"]
    if cat:
        zmid = min({r["z"] for r in cat}, key=lambda x: abs(x - 5.0))
        sample = [r for r in cat if r["z"] == zmid]
        weights = np.array([r["weight_cMpc3"] for r in sample])
        # Visualize draws from the predicted observed distribution, rather
        # than plotting the importance-sampling support uniformly.
        plot_rng = np.random.default_rng(90210)
        probability = weights / np.sum(weights)
        draw = plot_rng.choice(len(sample), size=min(1800, max(600, len(sample))),
                               replace=True, p=probability)
        plotted = [sample[j] for j in draw]
        fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.2))
        pairs = [
            ("lognh", "log_lx", r"$\log N_H$", r"$\log L_X$"),
            ("tau_e", "variability_rms", r"$\tau_e$", "fractional variability"),
            ("reff_pc", "fwhm_kms", r"$R_{\rm eff}$ [pc]", "Hα FWHM [km/s]"),
        ]
        for a, (xk, yk, xl, yl) in zip(ax, pairs):
            x = np.array([r[xk] for r in plotted])
            y = np.array([r[yk] for r in plotted])
            colour = np.log10(np.maximum([r["fedd"] for r in plotted], 1e-4))
            a.scatter(x, y, s=7, c=colour, cmap="viridis",
                      alpha=0.35, linewidth=0)
            a.set(xlabel=xl, ylabel=yl)
            xlo, xhi = np.quantile(x, [0.005, 0.995])
            ylo, yhi = np.quantile(y, [0.005, 0.995])
            if xhi > xlo:
                a.set_xlim(xlo, xhi)
            if yhi > ylo:
                a.set_ylim(ylo, yhi)
            a.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Predicted joint observables at z≈{zmid:.1f}", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUT / f"joint_observables_{suffix}.pdf")
        fig.savefig(OUT / f"joint_observables_{suffix}.png", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-halo", type=int, default=Config.n_halo)
    parser.add_argument("--dt-gyr", type=float, default=Config.dt_gyr)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--suffix", default="fiducial")
    parser.add_argument("--early-burst-rate", type=float,
                        default=Config.early_burst_rate_gyr,
                        help="early gas-rich compaction rate [Gyr^-1]")
    parser.add_argument("--early-burst-reservoir-fraction", type=float,
                        default=Config.early_burst_reservoir_fraction,
                        help="host baryon fraction deposited in a burst episode")
    parser.add_argument("--early-burst-visibility-gyr", type=float,
                        default=Config.early_burst_visibility_gyr,
                        help="reservoir depletion timescale [Gyr]")
    parser.add_argument("--porous-envelope", action="store_true",
                        help="apply the stochastic two-scale partial-covering reprocessor")
    parser.add_argument("--clump-transfer", action="store_true",
                        help="sample a clump-column transfer distribution inside the porous envelope")
    parser.add_argument("--cloudy-pilot", action="store_true",
                        help="replace the AGN H-alpha proxy with the screening-grade Cloudy emulator")
    parser.add_argument("--cloudy-two-zone", action="store_true",
                        help="mix dense pilot emission with the thermal diffuse skin")
    parser.add_argument("--physical-visibility", action="store_true",
                        help="predict dense Q(H), continuum subtype, and soft-EUV ratios from geometry")
    parser.add_argument("--coregulated-covering", action="store_true",
                        help="drive the envelope covering odds with the intrinsic engine/host contrast")
    parser.add_argument("--covering-odds-normalization", type=float,
                        default=Config.covering_odds_normalization,
                        help="covering odds at unit engine/host 2500 A contrast")
    parser.add_argument("--covering-odds-index", type=float,
                        default=Config.covering_odds_index,
                        help="power of the engine/host contrast in the covering odds")
    parser.add_argument("--dynamic-covering", action="store_true",
                        help="evolve covering through the build/clear ODE (equilibrium C/(1-C)=x^2)")
    parser.add_argument("--covering-saturating-law", action="store_true",
                        help="use the sequential two-stage odds (k x)^2/(1+2 k x) instead of k x^gamma")
    parser.add_argument("--covering-poisson-law", action="store_true",
                        help="use the Poisson double-blocking covering C=1-exp(-kx)(1+kx)")
    parser.add_argument("--covering-metallicity-index", type=float,
                        default=Config.covering_metallicity_index,
                        help="multiply covering odds by (Z/0.2 Zsun)^delta (Route-A discriminator)")
    parser.add_argument("--covering-relax-gyr", type=float,
                        default=Config.covering_relax_gyr,
                        help="covering relaxation timescale [Gyr] for the dynamic mode")
    parser.add_argument("--visibility-layer-gain", type=float,
                        default=Config.visibility_layer_gain,
                        help="effective clump-layer growth above the fixed covering threshold")
    parser.add_argument("--visibility-coupling-index", type=float,
                        default=Config.visibility_coupling_index,
                        help="global core/host coupling exponent for the dense Q(H) odds")
    args = parser.parse_args()
    cfg = Config(n_halo=args.n_halo, seed=args.seed, dt_gyr=args.dt_gyr,
                 early_burst_rate_gyr=args.early_burst_rate,
                 early_burst_reservoir_fraction=args.early_burst_reservoir_fraction,
                 early_burst_visibility_gyr=args.early_burst_visibility_gyr,
                 porous_envelope_enabled=(args.porous_envelope or args.physical_visibility),
                 clump_transfer_enabled=(args.clump_transfer or args.physical_visibility),
                 cloudy_pilot_enabled=(args.cloudy_pilot or args.cloudy_two_zone),
                 cloudy_two_zone_enabled=args.cloudy_two_zone,
                 physical_visibility_enabled=args.physical_visibility,
                 visibility_layer_gain=args.visibility_layer_gain,
                 visibility_coupling_index=args.visibility_coupling_index,
                 coregulated_covering_enabled=args.coregulated_covering,
                 covering_odds_normalization=args.covering_odds_normalization,
                 covering_odds_index=args.covering_odds_index,
                 dynamic_covering_enabled=args.dynamic_covering,
                 covering_relax_gyr=args.covering_relax_gyr,
                 covering_saturating_law=args.covering_saturating_law,
                 covering_poisson_law=args.covering_poisson_law,
                 covering_metallicity_index=args.covering_metallicity_index)
    result = simulate(cfg)
    (OUT / f"lifecycle_results_{args.suffix}.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "catalog"}, indent=2) + "\n"
    )
    save_catalog_csv(result["catalog"], OUT / f"synthetic_catalog_{args.suffix}.csv")
    plot_results(result, args.suffix)
    print("Semi-analytic lifecycle model")
    print(f"calibration factor = {result['normalization']:.4g}")
    print(f"demographic chi2 = {result['demographic_fit']['chi2']:.2f} "
          f"for {result['demographic_fit']['nominal_dof_after_one_normalization']} nominal dof")
    for row in sorted(result["summaries"], key=lambda r: -r["z"]):
        print(f"z={row['z']:.2f} n={row['predicted_density']:.3e} "
              f"logMh={row['median_logmh']:.2f} logNH={row['median_log_nh']:.2f} "
              f"R={row['median_reff_pc']:.1f}pc b={row['mean_halo_bias']:.2f}")


if __name__ == "__main__":
    main()
