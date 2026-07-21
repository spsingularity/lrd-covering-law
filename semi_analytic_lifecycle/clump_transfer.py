"""Stochastic clump-population transfer surrogate for porous LRD envelopes.

This is not a non-LTE solver.  It samples a distribution f(N_H, area, angle)
and produces bounded escape fractions for use in forward-model experiments.
The public API is intentionally small so it can later be replaced by a
Cloudy/radiative-transfer emulator with the same outputs.
"""
from __future__ import annotations

import numpy as np


def clump_transfer(nh_bulk, covering, rng, n_clumps=24, n_xray_layers=4,
                   open_column_fraction=1e-4, open_scatter_dex=.45,
                   clump_column_floor=5e24, clump_scatter_dex=.55):
    """Return area-averaged UV, H-alpha and compact-source X-ray transfer.

    Open rays sample a low-column distribution; covered rays sample dense
    clumps.  X-rays cross several statistically independent layers, reflecting
    their more compact emitting region.  The line result is an escape surrogate
    only: true level populations and line redistribution require non-LTE.
    """
    nh_bulk, covering = np.broadcast_arrays(np.asarray(nh_bulk, float), np.asarray(covering, float))
    shape = nh_bulk.shape
    cover = np.clip(covering, 0., 1.)
    open_prob = 1. - cover
    ray_open = rng.random(shape + (n_clumps,)) < open_prob[..., None]
    open_center = np.maximum(1e19, nh_bulk * open_column_fraction)
    clump_center = np.maximum(clump_column_floor, nh_bulk)
    open_cols = 10**rng.normal(np.log10(open_center)[..., None], open_scatter_dex, shape + (n_clumps,))
    dense_cols = 10**rng.normal(np.log10(clump_center)[..., None], clump_scatter_dex, shape + (n_clumps,))
    columns = np.where(ray_open, open_cols, dense_cols)
    # Effective cross sections are declared broadband surrogates, selected to
    # give transparent UV channels and strongly attenuated hard X-rays through
    # dense clumps.  They are not atomic cross-section tables.
    uv_trans = np.mean(np.exp(-np.minimum(30., 1e-22 * columns)), axis=-1)
    line_surface = np.mean(np.exp(-np.minimum(30., 6e-24 * columns)), axis=-1)
    balmer_escape = .05 + .95 * np.sqrt(line_surface)
    # A compact X-ray source intersects a stack of independently sampled clump
    # screens.  This is geometric partial covering rather than an imposed NH
    # floor, and is the particular assumption future imaging/transfer must test.
    ix = rng.integers(0, n_clumps, size=shape + (n_xray_layers,))
    stacked_columns = np.take_along_axis(columns, ix, axis=-1).sum(axis=-1)
    xray_trans = np.exp(-np.minimum(30., 2e-24 * stacked_columns))
    return {
        "uv_transmission": uv_trans,
        "balmer_escape": balmer_escape,
        "xray_transmission": xray_trans,
        "mean_channel_column": np.mean(columns, axis=-1),
        "xray_effective_column": stacked_columns,
        "open_area_fraction": np.mean(ray_open, axis=-1),
    }
