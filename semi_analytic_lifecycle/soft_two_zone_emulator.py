"""Thermal Cloudy emulator for the soft-EUV dense-plus-diffuse model.

The grid is deliberately restricted to the calculated 50, 70, and 100 kK
hardness bases and -2.5 <= log U_diffuse <= -1.5.  Temperatures label SED
hardness; they are not asserted to be literal source temperatures.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


HERE = Path(__file__).resolve().parent
RUNS = HERE / "cloudy_lrd_pilot" / "runs"
TEMPERATURES_K = np.array([50000.0, 70000.0, 100000.0])
DIFFUSE_LOGU = np.array([-2.5, -2.0, -1.5])


def _diffuse_name(temperature_k, logu):
    stem = f"soft_diffuse_{int(temperature_k) // 1000}k"
    suffix = {-2.5: "", -2.0: "_u_m2p0", -1.5: "_u_m1p5"}[float(logu)]
    return stem + suffix


def _load_yields(name):
    path = RUNS / name / "result.json"
    result = json.loads(path.read_text())
    if not result.get("status", {}).get("cloudy_exited_ok"):
        raise ValueError(f"Cloudy anchor {name!r} is not successful")
    return {key: float(value)
            for key, value in result["erg_per_ionizing_photon"].items()}


class SoftTwoZoneEmulator:
    def __init__(self):
        diffuse = {
            (temperature, logu): _load_yields(_diffuse_name(temperature, logu))
            for temperature in TEMPERATURES_K for logu in DIFFUSE_LOGU
        }
        dense = {
            temperature: _load_yields(f"soft_dense_{int(temperature) // 1000}k")
            for temperature in TEMPERATURES_K
        }
        line_sets = [set(value) for value in diffuse.values()]
        line_sets.extend(set(value) for value in dense.values())
        if any(lines != line_sets[0] for lines in line_sets[1:]):
            raise ValueError("soft Cloudy anchors do not contain identical lines")
        self.lines = tuple(sorted(line_sets[0]))
        self._diffuse_interpolators = {}
        self._dense_log_yields = {}
        for line in self.lines:
            grid = np.array([
                [np.log10(max(diffuse[(temperature, logu)][line], 1.0e-99))
                 for logu in DIFFUSE_LOGU]
                for temperature in TEMPERATURES_K
            ])
            self._diffuse_interpolators[line] = RegularGridInterpolator(
                (TEMPERATURES_K, DIFFUSE_LOGU), grid,
                bounds_error=True,
            )
            self._dense_log_yields[line] = np.log10([
                max(dense[temperature][line], 1.0e-99)
                for temperature in TEMPERATURES_K
            ])

    def predict(self, temperature_k, diffuse_logu, dense_fraction, clip=False):
        temperature, logu, fraction = np.broadcast_arrays(
            np.asarray(temperature_k, float),
            np.asarray(diffuse_logu, float),
            np.asarray(dense_fraction, float),
        )
        if np.any((fraction < 0.0) | (fraction > 1.0)):
            raise ValueError("dense_fraction must lie in [0, 1]")
        if clip:
            temperature = np.clip(temperature, TEMPERATURES_K[0], TEMPERATURES_K[-1])
            logu = np.clip(logu, DIFFUSE_LOGU[0], DIFFUSE_LOGU[-1])
        elif (np.any((temperature < TEMPERATURES_K[0]) |
                     (temperature > TEMPERATURES_K[-1])) or
              np.any((logu < DIFFUSE_LOGU[0]) | (logu > DIFFUSE_LOGU[-1]))):
            raise ValueError("temperature or log U lies outside soft-grid support")
        points = np.column_stack([temperature.ravel(), logu.ravel()])
        yields = {}
        for line in self.lines:
            diffuse = 10.0 ** self._diffuse_interpolators[line](points).reshape(
                temperature.shape
            )
            dense = 10.0 ** np.interp(
                temperature, TEMPERATURES_K, self._dense_log_yields[line]
            )
            yields[line] = fraction * dense + (1.0 - fraction) * diffuse
        return yields


@lru_cache(maxsize=1)
def get_soft_two_zone_emulator():
    return SoftTwoZoneEmulator()
