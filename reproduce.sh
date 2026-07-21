#!/usr/bin/env bash
# Single entry point: reproduce all results and build both papers.
#   Usage:  ./reproduce.sh
# Requires the dependencies in requirements.txt (pip install -r requirements.txt)
# and a LaTeX toolchain (pdflatex) for the paper step.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"

echo "==> [1/3] Running the analysis (results written into semi_analytic_lifecycle/)"
cd "$ROOT/semi_analytic_lifecycle"
$PY verify_equations_sympy.py             # 23/23 symbolic identity checks
$PY fit_public_lrd_stacks.py              # two-zone fit to the public stacks
$PY derive_interception_exponent.py       # Fig. 1
$PY refit_covering_ratio_space.py         # Table 1 ratios + covering-law scan (Table 2)
$PY plot_covering_scan.py                 # Fig. 2 (scan heatmap from the results)
$PY fit_two_zone_control.py               # Table 2 control
$PY confront_rubies_fwhm.py               # Fig. 3 (RUBIES widths)
$PY confront_rubies_fwhm_coregulated.py   # Fig. 3 coregulated variant
$PY analyze_subtype_march.py              # Fig. 4 + Letter Table 1 data
$PY plot_subtype_march_bands.py           # Letter Fig. 1 (bands)
$PY confront_marks.py                     # marks confrontation
$PY highz_marks.py                        # Letter Table 2 (z~9.5 gate-passing)
$PY evaluate_physical_visibility_closure.py

echo "==> [2/3] Collecting the manuscript figures into paper/"
cd "$ROOT"
cp semi_analytic_lifecycle/interception_exponent.pdf         paper/
cp semi_analytic_lifecycle/coregulated_covering_scan.pdf     paper/
cp semi_analytic_lifecycle/subtype_redshift_march.pdf        paper/
cp semi_analytic_lifecycle/subtype_march_with_bands.pdf      paper/
cp semi_analytic_lifecycle/public_lrd_constraints/rubies_fwhm_confrontation.pdf paper/

echo "==> [3/3] Compiling the papers"
cd "$ROOT/paper"
pdflatex -interaction=nonstopmode -halt-on-error ms_aastex.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error ms_aastex.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error letter_subtype_march.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error letter_subtype_march.tex >/dev/null

echo "==> DONE"
echo "    results : semi_analytic_lifecycle/  (JSON/CSV tables, figures)"
echo "    papers  : paper/ms_aastex.pdf , paper/letter_subtype_march.pdf"
