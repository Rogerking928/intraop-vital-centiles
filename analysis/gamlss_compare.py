"""GAMLSS（R，gamlss_fit.R）與主分析分位數迴歸的比較：20／50／80 歲的 P3、P50、P97，
以及兩種方法在參考族群內落在 P3 以下、P97 以下的比例（同樣是樣本內）。輸出 out/gamlss_compare.csv"""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, sys, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfit import fit_both, centile_of
P = _PROJ
g = pd.read_csv(f'{P}/out/gamlss_centiles.csv')
gf = pd.read_csv(f'{P}/out/gamlss_fit.csv')
q = pd.read_csv(f'{P}/out/norms_table.csv')
ref = pd.read_parquet(f'{P}/out/cohort_mover.parquet').query('ref')
rows = []
for v in g['var'].unique():
    hit = centile_of(fit_both(ref, v, [0.03, 0.5, 0.97]), ref, v).join(ref.sex)
    for s in ('Female', 'Male'):
        h = hit[hit.sex == s]
        f = gf[(gf['var'] == v) & (gf.sex == s)].iloc[0]
        for a in (20, 50, 80):
            x = q[(q['var'] == v) & (q.sex == s) & (q.age == a)].iloc[0]
            y = g[(g['var'] == v) & (g.sex == s) & (g.age == a)].iloc[0]
            rows.append({'var': v, 'sex': s, 'age': a, 'family': f.family,
                         'qr_P3': x.P3, 'qr_P50': x.P50, 'qr_P97': x.P97, 'g_P3': y.P3, 'g_P50': y.P50, 'g_P97': y.P97,
                         'qr_below_P3': 100 * h[0.03].mean(), 'qr_below_P97': 100 * h[0.97].mean(),
                         'g_below_P3': 100 * f.below_P3, 'g_below_P97': 100 * f.below_P97})
d = pd.DataFrame(rows)
d.to_csv(f'{P}/out/gamlss_compare.csv', index=False, float_format='%.12g')
d['dmax'] = d[['qr_P3', 'qr_P50', 'qr_P97']].values.__sub__(d[['g_P3', 'g_P50', 'g_P97']].values).__abs__().max(axis=1)
print(d.groupby('var').dmax.max().round(2)); print(d.drop_duplicates(['var','sex'])[['var','sex','family','qr_below_P3','g_below_P3','qr_below_P97','g_below_P97']].round(1))
