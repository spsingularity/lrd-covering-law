"""Conservative separable interpolator for the one-factor Cloudy pilot.

This is an observation-layer screening tool, not a multidimensional Cloudy
replacement. It exactly reproduces the successful one-axis pilot points and
adds their log responses when several parameters change simultaneously.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_GRID = HERE / "cloudy_lrd_pilot" / "pilot_grid.csv"
EXTENSION_GRID = HERE / "cloudy_lrd_pilot" / "extension_grid.csv"
OUTPUTS = (
    "yield_halpha", "yield_hbeta", "yield_oiii5007", "yield_nii6583",
    "yield_sii6716", "yield_sii6731", "yield_heii1640", "yield_heii4686",
    "yield_paalpha",
)
AXES = {
    "logn": ("density_low", "dense_n_9", "baseline", "dense_n_11", "density_high"),
    "logu": ("ionization_low", "baseline", "ionization_high",
             "dense_u_0p5", "dense_u_1p0"),
    "lognh": ("column_thin", "baseline"),
    "logz": ("dense_z_0p02", "metal_poor", "baseline", "metal_solar"),
}


class CloudyPilotEmulator:
    """Piecewise-linear log-response emulator inside the pilot bounds."""

    def __init__(self, path=DEFAULT_GRID):
        with Path(path).open() as handle:
            rows = list(csv.DictReader(handle))
        if EXTENSION_GRID.is_file():
            with EXTENSION_GRID.open() as handle:
                rows.extend(csv.DictReader(handle))
        self.rows = {row["model"]: row for row in rows}
        if "baseline" not in self.rows:
            raise ValueError("pilot grid has no baseline")
        self.baseline = self.rows["baseline"]

    @staticmethod
    def _numeric(row, key):
        return float(row[key])

    def _axis_points(self, axis, output):
        names = AXES[axis]
        coordinate = "metallicity" if axis == "logz" else axis
        x = np.array([self._numeric(self.rows[name], coordinate) for name in names])
        if axis == "logz":
            x = np.log10(x)
        base = self._numeric(self.baseline, output)
        axis_values = np.array([self._numeric(self.rows[name], output) for name in names])
        # Cloudy can print an exact zero for an undetectably weak line. Retain
        # it as a 30-dex upper-bound floor so interpolation stays finite.
        response = np.log10(np.maximum(axis_values, base * 1.0e-30) / base)
        order = np.argsort(x)
        return x[order], response[order]

    def predict(self, logn, logu, lognh, metallicity, sed="agn_standard",
                geometry="open", clip=False):
        """Predict line yields in erg per incident H-ionizing photon.

        Continuous inputs may be arrays. Values outside the pilot range raise
        by default; ``clip=True`` is available for explicitly bounded use in
        a population model.
        """
        values = np.broadcast_arrays(
            np.asarray(logn, dtype=float), np.asarray(logu, dtype=float),
            np.asarray(lognh, dtype=float), np.asarray(metallicity, dtype=float),
        )
        coordinates = {
            "logn": values[0], "logu": values[1], "lognh": values[2],
            "logz": np.log10(values[3]),
        }
        if np.any(values[3] <= 0):
            raise ValueError("metallicity must be positive")
        if sed not in {"agn_standard", "agn_xray_weak"}:
            raise ValueError("pilot emulator supports standard or X-ray-weak AGN SED")
        if geometry not in {"open", "sphere"}:
            raise ValueError("pilot emulator supports open or sphere geometry")

        prediction = {}
        for output in OUTPUTS:
            log_value = np.full(values[0].shape, np.log10(self._numeric(self.baseline, output)))
            for axis, coordinate_values in coordinates.items():
                xp, response = self._axis_points(axis, output)
                if not clip and (np.any(coordinate_values < xp[0]) or
                                 np.any(coordinate_values > xp[-1])):
                    raise ValueError(f"{axis} lies outside pilot range [{xp[0]}, {xp[-1]}]")
                log_value += np.interp(coordinate_values, xp, response)
            if sed == "agn_xray_weak":
                log_value += np.log10(
                    self._numeric(self.rows["xray_weak_sed"], output) /
                    self._numeric(self.baseline, output)
                )
            if geometry == "sphere":
                log_value += np.log10(
                    self._numeric(self.rows["closed_sphere"], output) /
                    self._numeric(self.baseline, output)
                )
            prediction[output.removeprefix("yield_")] = 10.0**log_value
        return prediction


@lru_cache(maxsize=1)
def get_pilot_emulator():
    return CloudyPilotEmulator()
