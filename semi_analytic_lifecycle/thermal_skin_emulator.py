"""Separable interpolator for the self-consistent thermal skin anchors.

This small emulator is intentionally restricted to the axes actually run:
log U at Z=0.2 and metallicity at log U=-2.  It is suitable for screening a
diffuse line-emitting phase, not for extrapolating a general Cloudy grid.
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_GRID = HERE / "cloudy_lrd_pilot" / "thermal_anchor_grid.csv"
RUNS = HERE / "cloudy_lrd_pilot" / "runs"
OUTPUTS = (
    "yield_halpha", "yield_hbeta", "yield_oiii5007", "yield_nii6583",
    "yield_sii6716", "yield_sii6731", "yield_heii1640", "yield_heii4686",
    "yield_paalpha",
)
AXES = {
    "logu": ("thermal_skin_u_low", "thermal_skin_u_mid", "thermal_skin_base"),
    "logz": ("thermal_skin_z_0p02", "thermal_skin_base", "thermal_skin_z_solar"),
}


class ThermalSkinEmulator:
    def __init__(self, path=DEFAULT_GRID):
        with Path(path).open() as handle:
            self.rows = {row["model"]: row for row in csv.DictReader(handle)}
        required = {name for names in AXES.values() for name in names}
        missing = required - self.rows.keys()
        if missing:
            raise ValueError(f"thermal anchor grid is missing {sorted(missing)}")
        self.baseline = self.rows["thermal_skin_base"]

    @staticmethod
    def _number(row, key):
        return float(row[key])

    def _axis(self, name, output):
        coordinate = "metallicity" if name == "logz" else name
        x = np.array([self._number(self.rows[key], coordinate) for key in AXES[name]])
        if name == "logz":
            x = np.log10(x)
        baseline = self._number(self.baseline, output)
        y = np.log10(np.array([
            self._number(self.rows[key], output) for key in AXES[name]
        ]) / baseline)
        order = np.argsort(x)
        return x[order], y[order]

    def predict(self, logu, metallicity, clip=False):
        logu, metallicity = np.broadcast_arrays(
            np.asarray(logu, dtype=float), np.asarray(metallicity, dtype=float)
        )
        if np.any(metallicity <= 0):
            raise ValueError("metallicity must be positive")
        coordinates = {"logu": logu, "logz": np.log10(metallicity)}
        prediction = {}
        for output in OUTPUTS:
            log_value = np.full(logu.shape,
                                np.log10(self._number(self.baseline, output)))
            for axis, coordinate in coordinates.items():
                xp, response = self._axis(axis, output)
                if not clip and (np.any(coordinate < xp[0]) or np.any(coordinate > xp[-1])):
                    raise ValueError(f"{axis} lies outside thermal anchor support "
                                     f"[{xp[0]}, {xp[-1]}]")
                log_value += np.interp(coordinate, xp, response)
            prediction[output.removeprefix("yield_")] = 10.0**log_value
        return prediction


@lru_cache(maxsize=1)
def get_thermal_skin_emulator():
    return ThermalSkinEmulator()


@lru_cache(maxsize=1)
def get_dense_thermal_correction():
    """Return the thermal/fixed yield correction at the dense skin anchor.

    Both models have log n=10, log U=-1, log N_H=21, and Z=0.2.  Applying
    this correction away from that anchor remains a screening approximation,
    but it removes the measured fixed-temperature bias exactly at the anchor.
    """
    thermal = json.loads((RUNS / "thermal_dense_skin" / "result.json").read_text())
    fixed = json.loads((RUNS / "column_thin" / "result.json").read_text())
    thermal_yields = thermal["erg_per_ionizing_photon"]
    fixed_yields = fixed["erg_per_ionizing_photon"]
    return {line: thermal_yields[line] / fixed_yields[line]
            for line in thermal_yields}
