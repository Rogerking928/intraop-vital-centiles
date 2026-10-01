"""讀值層級指標的參考百分位（參考族群 ASA I–II），回應「中位數與單筆讀值不能比」與「排除誘導期」：
  maint_sust5：維持期持續 5 分鐘的最低 MAP；maint_twa：維持期時間加權平均 MAP；
  full_median／full_sust5：含誘導期（整段麻醉）的中位數與持續 5 分鐘最低 MAP。
輸出
  out/norms2_table.csv      每一歲、每性別、四個指標的 P3–P97
  out/norms2_dec.csv        每 10 歲 P3/P50/P97，並列主分析（維持期中位數）
  out/threshold2.csv        65 mmHg 落在各指標分布的第幾百分位（年齡段中點）
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
