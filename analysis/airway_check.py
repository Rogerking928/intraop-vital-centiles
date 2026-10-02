"""Sensitivity check for the end-tidal CO2 centiles (added after external review, 2026-10-02): refit them without
reference patients whose only intraoperative oxygen device recorded was not an advanced airway (nasal cannula, mask,
room air). Uses out/mover_airway_raw.parquet from extract_airway.py. Output out/airway_check.json."""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, re, json, numpy as np, pandas as pd, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfit import fit_both, predict

P = _PROJ
a = pd.read_parquet(f'{P}/out/mover_airway_raw.parquet')


def kind(vals):
    t = ' ; '.join(vals).upper()
    if re.search(r'\bETT\b|ENDOTRACH|TRACH', t): return 'advanced'
    if re.search(r'LMA|LARYNGEAL|SUPRAGLOTT', t): return 'advanced'
    if re.search(r'NASAL|CANNULA|MASK|ROOM AIR|BLOW|\bNC\b', t): return 'none'
    return 'other'


k = a.groupby('LOG_ID').v.apply(lambda s: kind(s.dropna().astype(str)))
ref = pd.read_parquet(f'{P}/out/cohort_mover.parquet').query('ref')
ref['airway'] = ref.LOG_ID.map(k).fillna('not recorded')
sub = ref[ref.airway != 'none']
AGES = [20, 30, 40, 50, 60, 70, 80, 90]
out = {'reference': int(len(ref)), 'recorded': int((ref.airway != 'not recorded').sum()),
       'no_advanced_airway': int((ref.airway == 'none').sum()),
       'etco2_lt30_pct_no_advanced': float(100 * (ref[ref.airway == 'none'].etco2 < 30).mean()),
       'etco2_lt30_pct_rest': float(100 * (ref[ref.airway != 'none'].etco2 < 30).mean()), 'max_abs_change': {}}
for s, m in fit_both(ref, 'etco2').items():
    m2 = fit_both(sub, 'etco2')[s]
    p1, p2 = predict(m, AGES), predict(m2, AGES)
    out['max_abs_change'][s] = {f'P{round(q * 100)}': float((p2[q] - p1[q]).abs().max()) for q in p1.columns}
json.dump(out, open(f'{P}/out/airway_check.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
