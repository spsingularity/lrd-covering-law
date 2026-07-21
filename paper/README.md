# Manuscript sources

LaTeX sources (AASTeX v6.31) for the main paper (`ms_aastex.tex`) and the
companion Letter (`letter_subtype_march.tex`). Bibliographies are inline; no
external `.bib` is needed.

Figures are **not** committed — they are produced by the analysis and copied
here. The top-level `../reproduce.sh` does this and then compiles both papers.
To build the papers alone (after the analysis has been run at least once):

```bash
cp ../semi_analytic_lifecycle/{interception_exponent,coregulated_covering_scan,subtype_redshift_march,subtype_march_with_bands}.pdf .
cp ../semi_analytic_lifecycle/public_lrd_constraints/rubies_fwhm_confrontation.pdf .
pdflatex ms_aastex.tex && pdflatex ms_aastex.tex
pdflatex letter_subtype_march.tex && pdflatex letter_subtype_march.tex
```
