#!/usr/bin/env python3
"""RUBIES width confrontation with the co-regulated covering law enabled.

Thin wrapper around ``confront_rubies_fwhm.py``: identical data, matching,
radius scan, bootstrap, and outputs, with two controlled changes only:

1. the lifecycle covering law is the scan-preferred co-regulated form
   (covering odds equal to the squared engine/host 2500 A contrast);
2. outputs are written to ``public_lrd_constraints/coregulated/`` so the
   published confrontation record is not overwritten.

This checks that the covering revision does not break the compact
broad-line-scale agreement, which depends on the selected population's
black-hole masses and Thomson depths.
"""

from __future__ import annotations

from pathlib import Path

import confront_rubies_fwhm as base
from lrd_lifecycle import Config

HERE = Path(__file__).resolve().parent


class CoregulatedConfig(Config):
    """The paper's fitted-law model: co-regulated covering with the
    physical-visibility observation layer and the fiducial integration step
    (the same configuration that produces the coregulated_covering_mid
    catalogue used for Tables 1-2, Fig. 1, and the demography)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("coregulated_covering_enabled", True)
        kwargs.setdefault("covering_odds_index", 2.0)
        kwargs.setdefault("covering_odds_normalization", 1.0)
        kwargs.setdefault("physical_visibility_enabled", True)
        kwargs["dt_gyr"] = 0.006
        super().__init__(**kwargs)


def main() -> None:
    base.Config = CoregulatedConfig
    base.OUT = HERE / "public_lrd_constraints" / "coregulated"
    # The co-regulated selection shifts the descriptive radius upward past
    # the published 5000 AU scan edge; widen the support so the fit and its
    # bootstrap interval are interior rather than railed.
    base.RADIUS_GRID_AU = (250.0, 40000.0, 601)
    base.main()


if __name__ == "__main__":
    main()
