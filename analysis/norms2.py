"""Reference centiles for reading-level metrics (reference population ASA I-II), addressing review comments that
"a case median cannot be compared with single readings" and "induction is excluded":
  maint_sust5: lowest 5-min sustained MAP in maintenance; maint_twa: time-weighted average MAP in maintenance;
  full_median/full_sust5: median and lowest 5-min sustained MAP including induction (whole anaesthetic).
Outputs
  out/norms2_table.csv      P3-P97 for every year of age, each sex, the four metrics
  out/norms2_dec.csv        P3/P50/P97 every 10 years, alongside the primary analysis (maintenance median)
  out/threshold2.csv        centile of 65 mmHg in each metric's distribution (age-band midpoints)
"""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfit import fit_both, predict

P = _PROJ
mv = pd.read_parquet(f'{P}/out/cohort_mover.parquet').merge(pd.read_parquet(f'{P}/out/readings_mover.parquet'), on='LOG_ID')
ref = mv[mv.ref].copy()
M = {'nibp_map': 'Maintenance, case median (primary)', 'maint_twa': 'Maintenance, time-weighted average',
     'maint_sust5': 'Maintenance, lowest sustained 5 min', 'full_median': 'Whole anaesthesia incl. induction, case median',
     'full_sust5': 'Whole anaesthesia incl. induction, lowest sustained 5 min'}
AGES = np.arange(18, 91)
rows, dec = [], []
for v in M:
    for s, m in fit_both(ref, v).items():
        pr = predict(m, AGES)
        for a in AGES:
            r = {'metric': v, 'sex': s, 'age': int(a), **{f'P{int(round(q*100))}': pr.at[a, q] for q in pr.columns}}
            rows.append(r)
            if a % 10 == 0: dec.append(r)
pd.DataFrame(rows).to_csv(f'{P}/out/norms2_table.csv', index=False, float_format='%.12g')
pd.DataFrame(dec).to_csv(f'{P}/out/norms2_dec.csv', index=False, float_format='%.12g')

qs = [round(x, 3) for x in np.arange(0.005, 0.9951, 0.005)]
MID = {'18-29': 24, '30-39': 35, '40-49': 45, '50-59': 55, '60-69': 65, '70-79': 75, '80-90': 85}
thr = []
for v in ['nibp_map', 'maint_sust5', 'full_sust5']:
    for s, m in fit_both(ref, v, qs).items():
        pr = predict(m, list(MID.values()))
        for b, a in MID.items():
            thr.append({'metric': v, 'sex': s, 'band': b,
                        'centile_of_65': 100 * np.interp(65, pr.loc[a].values, qs, left=0.0, right=1.0)})
t = pd.DataFrame(thr).pivot_table(index=['sex', 'band'], columns='metric', values='centile_of_65').reset_index()
ref['band'] = pd.cut(ref.age, [18, 30, 40, 50, 60, 70, 80, 91], right=False, labels=list(MID))
e = ref.groupby(['sex', 'band'], observed=True).agg(
    pct_sust5_lt65=('maint_sust5', lambda x: 100 * (x < 65).mean()),
    pct_full_sust5_lt65=('full_sust5', lambda x: 100 * (x < 65).mean()),
    pct_full_min1_lt65=('full_min1', lambda x: 100 * (x < 65).mean())).reset_index()
t = t.merge(e, on=['sex', 'band'])
t.to_csv(f'{P}/out/threshold2.csv', index=False, float_format='%.12g')
import json
json.dump({'pct_sust5_lt65': 100 * float((ref.maint_sust5 < 65).mean()), 'n': int(len(ref))},
          open(f'{P}/out/threshold2_overall.json', 'w'), indent=1)
pd.set_option('display.width', 220)
print(pd.DataFrame(dec).query("age in (20,50,80)")[['metric', 'sex', 'age', 'P3', 'P50', 'P97']].round(1).to_string())
print(t.round(1).to_string())
