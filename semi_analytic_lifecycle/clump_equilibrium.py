"""Time-resolved clump-covering dynamics behind the x^2 law.

Derivation (see the manuscript, Sec. 3).  The porous envelope covering C
evolves by competition between an envelope-building rate per uncovered solid
angle, b, and a clearing rate per covered solid angle, g:

    dC/dt = (1 - C) b - C g .

The candidate mechanism gives each rate one power of the intrinsic
engine/host contrast x:

    b = nu * x      (clump condensation supplied by engine feeding),
    g = nu / x      (host-powered ablation resisted by engine confinement),

with a single micro-rate nu shared by both processes (the "same gas, same
site" assumption).  The equilibrium is then exactly the scan-preferred law

    C / (1 - C) = b / g = x^2 ,   C_eq = x^2 / (1 + x^2) ,

with the parity anchor C = 1/2 at x = 1 following from the shared nu.  The
relaxation toward equilibrium has total rate r = b + g = nu (x + 1/x), so
covering lags the engine whenever the engine evolves faster than 1/r: young
or rapidly changing sources sit off the equilibrium law (hysteresis), which
is the falsifiable difference between this dynamical model and the
stationary x^2 closure.

Because the ODE is linear in C at frozen coefficients, each timestep is
integrated exactly:

    C(t + dt) = C_eq + (C - C_eq) exp(-r dt) .
"""
from __future__ import annotations

import numpy as np


def equilibrium_covering(contrast):
    """C_eq = x^2 / (1 + x^2), the stationary limit of the dynamics."""
    x2 = np.clip(np.asarray(contrast, float), 1.0e-12, 1.0e12) ** 2
    return x2 / (1.0 + x2)


def relaxation_rate_gyr(contrast, relax_gyr):
    """Total relaxation rate r = nu (x + 1/x), with nu = 1/relax_gyr."""
    x = np.clip(np.asarray(contrast, float), 1.0e-12, 1.0e12)
    return (x + 1.0 / x) / max(float(relax_gyr), 1.0e-6)


def step_covering(cover, contrast, relax_gyr, dt_gyr):
    """Advance covering one timestep with the exact linear-ODE update."""
    cover = np.asarray(cover, float)
    c_eq = equilibrium_covering(contrast)
    rate = relaxation_rate_gyr(contrast, relax_gyr)
    decay = np.exp(-np.minimum(rate * float(dt_gyr), 60.0))
    return np.clip(c_eq + (cover - c_eq) * decay, 0.0, 1.0)
