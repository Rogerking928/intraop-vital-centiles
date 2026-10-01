"""VitalDB 外部驗證：MOVER 參考曲線（ASA I–II）套到 VitalDB（ASA I–II、非急診）。

每個變數：落在 MOVER 同年齡×性別 P3/P10/P50/P90/P97 以下的比例（Wilson 95% CI），
整體、分性別、分年齡段、分科別、排除升壓劑輸注、排除有動脈導管者（ART 訊號 ≤20 分鐘）；另報位置偏移（VitalDB 值減 MOVER P50 的中位數，bootstrap CI）。
輸出 out/validation.csv、out/validation_shift.csv
"""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfit import Q, fit_both, predict, centile_of
from statsmodels.stats.proportion import proportion_confint

P = _PROJ
V = ['nibp_map', 'nibp_sbp', 'nibp_dbp', 'hr', 'etco2', 'temp_c']
BANDS = [18, 30, 40, 50, 60, 70, 80, 91]
BLAB = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-90']
mv = pd.read_parquet(f'{P}/out/cohort_mover.parquet')
ref = mv[mv.ref]
vd = pd.read_parquet(f'{P}/out/cohort_vitaldb.parquet')
vd = vd[vd.ref].copy()
# MOVER 年齡截在 90、參考曲線只到 90：VitalDB 也截在 90（原本 >90 的人被拿去比外推的百分位，且不在任何年齡層）
vd['age'] = vd.age.clip(upper=90)
for k in ['nibp_map', 'nibp_sbp', 'nibp_dbp']:
    vd.loc[vd.n_nibp < 3, k] = np.nan
vd['band'] = pd.cut(vd.age, BANDS, right=False, labels=BLAB)

rows, shifts = [], []
rng = np.random.default_rng(7)
for v in V:
    models = fit_both(ref, v)
    hit = centile_of(models, vd, v).join(vd[['sex', 'band', 'department', 'vaso_inf', 'n_art']])
    groups = [('All', 'All', hit)]
    groups += [('Sex', s, g) for s, g in hit.groupby('sex')]
    groups += [('Age', b, g) for b, g in hit.groupby('band', observed=True)]
    groups += [('Department', d, g) for d, g in hit.groupby('department')]
    groups += [('No vasoactive infusion', 'All', hit[hit.vaso_inf == 0])]
    groups += [('No arterial line', 'All', hit[hit.n_art.fillna(0) <= 20])]
    for kind, lvl, g in groups:
        r = {'var': v, 'stratum': kind, 'level': lvl, 'n': len(g)}
        for q in Q:
            k = int(g[q].sum())
            lo, hi = proportion_confint(k, len(g), method='wilson')
            r[f'below_P{int(round(q*100))}'] = 100 * k / len(g)
            r[f'below_P{int(round(q*100))}_lo'], r[f'below_P{int(round(q*100))}_hi'] = 100 * lo, 100 * hi
        rows.append(r)
    # 位置偏移
    d = vd[['age', 'sex', v, 'department']].dropna()
    p50 = pd.concat([predict(models[s], d[d.sex == s].age.values)[0.5].set_axis(d[d.sex == s].index) for s in models])
    res = d[v] - p50.loc[d.index]
    for lvl, rr in [('All', res)] + [(dep, res[d.department == dep]) for dep in d.department.unique()]:
        bs = [np.median(rng.choice(rr.values, len(rr))) for _ in range(1000)]
        shifts.append({'var': v, 'level': lvl, 'n': len(rr), 'median_shift': np.median(rr),
                       'lo': np.percentile(bs, 2.5), 'hi': np.percentile(bs, 97.5)})

out = pd.DataFrame(rows)
out.to_csv(f'{P}/out/validation.csv', index=False, float_format='%.4f')
sh = pd.DataFrame(shifts)
sh.to_csv(f'{P}/out/validation_shift.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 250)
cols = ['var', 'stratum', 'level', 'n', 'below_P3', 'below_P10', 'below_P50', 'below_P90', 'below_P97']
print(out[cols].round(1).to_string())
print(sh.round(2).to_string())
