"""Input for gamlss_fit.R: the reference cohort (ASA I-II) with sex, age and the case medians of each vital sign.
Output out/ref_for_gamlss.csv. Run after cohort.py; then gamlss_fit.R, then gamlss_compare.py."""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, pandas as pd

P = _PROJ
m = pd.read_parquet(f'{P}/out/cohort_mover.parquet')
r = m[m.ref][['sex', 'age', 'nibp_map', 'nibp_sbp', 'nibp_dbp', 'hr', 'etco2', 'temp_c']]
r.to_csv(f'{P}/out/ref_for_gamlss.csv', index=False)
print(len(r))
