"""Run a one-factor-at-a-time Cloudy pilot around an LRD clump baseline."""
from __future__ import annotations

import csv
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from cloudy_bridge import CloudyParameters, photon_normalized_yields, run_model


HERE = Path(__file__).resolve().parent
OUT = HERE / "cloudy_lrd_pilot"
RUNS = OUT / "runs"


def pilot_models():
    """Return a bounded pilot spanning the highest-value physical axes."""
    base = CloudyParameters(
        logn=10.0,
        logu=-1.0,
        # The photoionized emitting skin is distinct from the much larger
        # obscuring sightline column handled by the clump-transfer layer.
        lognh=22.0,
        metallicity=0.2,
        dust_to_metals=0.0,
        sed="agn_standard",
        geometry="open",
        # A fast non-LTE screening tier. Thermal-balance anchors are retained
        # separately because the first dense anchor exceeded six minutes.
        temperature=1.0e4,
        no_molecules=True,
        iterations=3,
    )
    return {
        "baseline": base,
        "density_low": replace(base, logn=8.0),
        "density_high": replace(base, logn=12.0),
        "ionization_low": replace(base, logu=-2.5),
        "ionization_high": replace(base, logu=0.0),
        "column_thin": replace(base, lognh=21.0),
        "metal_poor": replace(base, metallicity=0.05),
        "metal_solar": replace(base, metallicity=1.0),
        "dusty": replace(base, dust_to_metals=1.0),
        "xray_weak_sed": replace(base, sed="agn_xray_weak"),
        "closed_sphere": replace(base, geometry="sphere", radius_log_cm=17.0),
    }


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else math.nan


def diagnostics(yields):
    sii = yields["sii6716"] + yields["sii6731"]
    result = {
        "halpha_hbeta": safe_ratio(yields["halpha"], yields["hbeta"]),
        "oiii_hbeta": safe_ratio(yields["oiii5007"], yields["hbeta"]),
        "nii_halpha": safe_ratio(yields["nii6583"], yields["halpha"]),
        "sii_halpha": safe_ratio(sii, yields["halpha"]),
        "heii1640_hbeta": safe_ratio(yields["heii1640"], yields["hbeta"]),
        "heii4686_hbeta": safe_ratio(yields["heii4686"], yields["hbeta"]),
        "paalpha_halpha": safe_ratio(yields["paalpha"], yields["halpha"]),
        "lha_per_qh54": yields["halpha"] * 1.0e54,
    }
    # These diagnostics were added after the first pilot.  Keep cache reads
    # backward compatible while exposing them for every new Cloudy run.
    if "hgamma" in yields:
        result["hbeta_hgamma"] = safe_ratio(yields["hbeta"], yields["hgamma"])
    if {"oiii4363", "oiii5007"} <= yields.keys():
        result["oiii5007_oiii4363"] = safe_ratio(
            yields["oiii5007"], yields["oiii4363"]
        )
    if {"oii3726", "oii3729"} <= yields.keys():
        oii = yields["oii3726"] + yields["oii3729"]
        result["oiii5007_oii3727"] = safe_ratio(yields["oiii5007"], oii)
        if "neiii3869" in yields:
            result["neiii3869_oii3727"] = safe_ratio(yields["neiii3869"], oii)
    if "oi6300" in yields:
        result["oi6300_halpha"] = safe_ratio(yields["oi6300"], yields["halpha"])
    if "hei5876" in yields:
        result["hei5876_hbeta"] = safe_ratio(yields["hei5876"], yields["hbeta"])
    if "hei7065" in yields:
        result["hei7065_hbeta"] = safe_ratio(yields["hei7065"], yields["hbeta"])
    return result


def run_one(name, params):
    destination = RUNS / name
    cached = destination / "result.json"
    if cached.is_file():
        result = json.loads(cached.read_text())
        if (result.get("status", {}).get("cloudy_exited_ok")
                and result.get("parameters") == params.__dict__):
            # Recompute derived quantities so cache files remain correct if a
            # normalization formula is repaired without rerunning Cloudy.
            result["erg_per_ionizing_photon"] = photon_normalized_yields(
                result["absolute_lines"], params
            )
            result["diagnostics"] = diagnostics(result["erg_per_ionizing_photon"])
            cached.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            result["cached"] = True
            return name, result
    failed_cache = destination / "failure.json"
    if failed_cache.is_file():
        failure = json.loads(failed_cache.read_text())
        if failure.get("parameters") == params.__dict__:
            failure["cached"] = True
            return name, failure
    started = time.monotonic()
    try:
        result = run_model(destination, params, timeout=360)
        result["elapsed_seconds"] = time.monotonic() - started
        result["cached"] = False
        result["diagnostics"] = diagnostics(result["erg_per_ionizing_photon"])
        cached.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return name, result
    except Exception as error:
        failure = {
            "parameters": params.__dict__,
            "status": {"cloudy_exited_ok": False},
            "elapsed_seconds": time.monotonic() - started,
            "error": f"{type(error).__name__}: {error}",
        }
        (destination / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
        return name, failure


def flatten(name, result):
    row = {"model": name, **result["parameters"], **result["status"]}
    row.update(result.get("diagnostics") or diagnostics(result["erg_per_ionizing_photon"]))
    row.update({f"yield_{key}": value
                for key, value in result["erg_per_ionizing_photon"].items()})
    row["elapsed_seconds"] = result.get("elapsed_seconds", math.nan)
    return row


def main(max_workers=2):
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    models = pilot_models()
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, name, params): name
                   for name, params in models.items()}
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
            print(name, "cached" if result.get("cached") else
                  ("complete" if result["status"]["cloudy_exited_ok"] else "FAILED"),
                  result["status"], flush=True)

    completed_names = [name for name in models
                       if results[name]["status"]["cloudy_exited_ok"]]
    failed_names = [name for name in models if name not in completed_names]
    if "baseline" not in completed_names:
        raise RuntimeError("baseline Cloudy model failed; pilot cannot be summarized")
    rows = [flatten(name, results[name]) for name in completed_names]
    with (OUT / "pilot_grid.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    baseline = rows[0]
    comparison_keys = (
        "lha_per_qh54", "halpha_hbeta", "oiii_hbeta", "nii_halpha",
        "sii_halpha", "heii1640_hbeta", "paalpha_halpha",
    )
    sensitivities = {}
    for row in rows[1:]:
        sensitivities[row["model"]] = {
            key: (math.log10(row[key] / baseline[key])
                  if row[key] > 0 and baseline[key] > 0 else math.nan)
            for key in comparison_keys
        }
    summary = {
        "scope": (
            "Fixed-temperature, three-iteration non-LTE Cloudy C25 screening "
            "pilot. This is a sensitivity design, not a thermal-balance "
            "posterior or a complete factorial grid."
        ),
        "normalization": (
            "Line energies per incident H-ionizing photon; lha_per_qh54 is "
            "the emergent H-alpha luminosity for Q(H)=1e54 s^-1 and unit covering."
        ),
        "baseline": baseline,
        "log10_change_from_baseline": sensitivities,
        "models_completed": len(rows),
        "models_failed": failed_names,
        "models_with_cautions": [row["model"] for row in rows if row["cautions"] > 0],
    }
    (OUT / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    print(json.dumps({"models_completed": len(rows), "models_failed": failed_names,
                      "models_with_cautions": summary["models_with_cautions"]}, indent=2))


if __name__ == "__main__":
    main()
