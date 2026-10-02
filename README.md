# Reference centiles for vital signs during general anaesthesia in adults

Age- and sex-specific reference centiles for non-invasive mean, systolic and diastolic arterial pressure, heart rate,
end-tidal CO2 and temperature during the maintenance phase of general anaesthesia in adults with ASA physical status
I-II, developed in MOVER (University of California, Irvine, USA) and externally validated in VitalDB (Seoul National
University Hospital, Korea).

**Calculator:** https://rogerking928.github.io/intraop-vital-centiles/ (the same page is `docs/index.html`; it runs in
the browser and sends nothing anywhere).

These centiles describe usual anaesthetic practice in healthy adults. They are not thresholds for harm. In the
external validation, heart rate and end-tidal CO2 centiles were miscalibrated and should be recalibrated locally.

## Contents

- `centiles/` - the reference centiles (P3, P10, P25, P50, P75, P90, P97) for every year of age from 18 to 90, by sex:
  `centiles_primary.csv` (patient median over maintenance), `centiles_no_vasoactive_drug.csv`, and
  `centiles_other_summaries.csv` (time-weighted average, lowest sustained 5-minute MAP, whole anaesthesia).
  These are aggregate model outputs; no patient-level data are included.
- `analysis/` - the complete analysis, in the order it is run:
  `extract_mover.py`, `extract_vitaldb.py`, `extract_preop.py`, `extract_meds.py`, `cohort.py`, `norms.py`,
  `validate.py`, `describe.py`, `readings.py`, `norms2.py`, `revision.py`, `outcome.py`, `outcome2.py`,
  `gamlss_input.py`, `gamlss_fit.R`, `gamlss_compare.py`, then the checks added after external review:
  `readings_gapcheck.py`, `extract_airway.py`, `airway_check.py` (`qfit.py` holds the shared quantile-regression code). Run Python scripts with `OPENBLAS_NUM_THREADS=1`; `qfit.py` also limits BLAS to one thread
  through threadpoolctl, without which quantile regression on tied (integer) data can differ by 0.1 between runs.
- `docs/index.html` - the calculator page (built by `figures/make_calculator.py`).
- `figures/figs.py` - the figures; `figures/make_calculator.py` - builds the calculator page.

## Data

The data are not redistributed. MOVER is available to registered users under a data use agreement from
https://mover.ics.uci.edu; VitalDB is openly available from https://vitaldb.net. Set `MOVER_DIR`, `VITALDB_DIR` and,
optionally, `PROJECT_DIR` (where intermediate files are written, default the repository root) before running.

## Software

Python 3.12.3; exact package versions in `requirements.txt`. R 4.4.3 with gamlss 5.5.0.

The analysis was written and run by the author in Python 3, with Claude (Anthropic) and ChatGPT (OpenAI) used as
assistants under the author's direction; the author reviewed and tested all code and takes full responsibility for
the results it produces.

## Licence

Code: MIT. Centile tables: CC BY 4.0.
