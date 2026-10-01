"""回應外部審閱（2026-09-30）的補充分析。
 A. 沒用任何升壓劑者的完整百分位表（次要參考）         → out/norms_table_nodrug.csv
 B. rearrangement 之前分位數曲線有沒有交叉               → out/crossing.json
 C. VitalDB 再校正示範：一半估偏移（與尺度），另一半驗證   → out/recalibration.csv
 D. VitalDB 分層：手術方式、TIVA、BMI 三分位、呼吸道      → out/validation_strata2.csv
 E. MOVER 共變項：BMI、腹腔鏡／機器手臂、丙泊酚輸注（TIVA）對 P3／P50／P97 的影響 → out/covariates_qr.csv
 F. 沒用升壓劑女性 EtCO2 低尾的來源：只看麻醉 ≥120 分  → out/etco2_nodrug_check.csv
"""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, sys, json, duckdb, numpy as np, pandas as pd, patsy, warnings
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfit import Q, fit, fit_both, predict, centile_of
warnings.simplefilter('ignore')

P = _PROJ
VD = _VITALDB
V = ['nibp_map', 'nibp_sbp', 'nibp_dbp', 'hr', 'etco2', 'temp_c']
AGES = np.arange(18, 91)
mv = pd.read_parquet(f'{P}/out/cohort_mover.parquet')
ref = mv[mv.ref].copy()
nodrug = ref[(ref.vaso_inf == 0) & (ref.vaso_bolus == 0)]

# ---- A
rows = []
for v in V:
    for s, m in fit_both(nodrug, v).items():
        pr = predict(m, AGES)
        for a in AGES:
            rows.append({'var': v, 'sex': s, 'age': int(a), **{f'P{int(round(q*100))}': pr.at[a, q] for q in pr.columns}})
pd.DataFrame(rows).to_csv(f'{P}/out/norms_table_nodrug.csv', index=False, float_format='%.4f')

# ---- B：未排序的預測有沒有交叉
cross = {}
for v in V:
    for s in ('Female', 'Male'):
        info, pars = fit(ref[ref.sex == s], v)
        X = np.asarray(patsy.build_design_matrices([info], {'age': AGES.astype(float)})[0])
        raw = np.column_stack([X @ pars[q] for q in sorted(pars)])
        cross[f'{v}|{s}'] = int((np.diff(raw, axis=1) < 0).any(axis=1).sum())
json.dump({'ages_with_crossing_before_rearrangement': cross, 'ages_checked': int(len(AGES))},
          open(f'{P}/out/crossing.json', 'w'), indent=1)

# ---- C：再校正
vd = pd.read_parquet(f'{P}/out/cohort_vitaldb.parquet')
vd = vd[vd.ref].copy()
vd['age'] = vd.age.clip(upper=90)   # 同 validate.py：MOVER 年齡截在 90
for k in ['nibp_map', 'nibp_sbp', 'nibp_dbp']:
    vd.loc[vd.n_nibp < 3, k] = np.nan
rng = np.random.default_rng(3)
vd['half'] = rng.integers(0, 2, len(vd))
QS = [0.03, 0.10, 0.50, 0.90, 0.97]
rec = []
for v in V:
    mods = fit_both(ref, v)
    d = vd[['age', 'sex', v, 'half']].dropna()
    pr = pd.concat([predict(mods[s], d[d.sex == s].age.values).set_axis(d[d.sex == s].index) for s in mods]).loc[d.index]
    resid = d[v] - pr[0.5]
    A, B = d.half == 0, d.half == 1
    shift = float(np.median(resid[A]))
    # 尺度：VitalDB-A 的 P10–P90 殘差寬度 ÷ MOVER 在同樣年齡×性別的 P10–P90 寬度（平均）
    k = float((np.quantile(resid[A], 0.9) - np.quantile(resid[A], 0.1)) / (pr.loc[A, 0.9] - pr.loc[A, 0.1]).mean())
    for method in ('None', 'Shift', 'Shift and scale'):
        r = {'var': v, 'method': method, 'n_estimate': int(A.sum()), 'n_test': int(B.sum()),
             'shift': shift if method != 'None' else 0.0, 'scale': k if method == 'Shift and scale' else 1.0}
        for q in QS:
            c = pr.loc[B, q]
            if method == 'Shift': c = c + shift
            if method == 'Shift and scale': c = pr.loc[B, 0.5] + shift + k * (pr.loc[B, q] - pr.loc[B, 0.5])
            r[f'below_P{int(round(q*100))}'] = 100 * float((d.loc[B, v] < c).mean())
        r['mean_abs_error'] = float(np.mean([abs(r[f'below_P{int(round(q*100))}'] - 100 * q) for q in QS]))
        rec.append(r)
pd.DataFrame(rec).to_csv(f'{P}/out/recalibration.csv', index=False, float_format='%.5f')

# ---- D：VitalDB 分層
vc = pd.read_csv(f'{VD}/cases.csv').set_index('caseid')
trk = pd.read_csv(f'{VD}/trks.csv')
tiva_ids = set(trk[trk.tname == 'Orchestra/PPF20_RATE'].caseid)
vd['approach'] = vd.caseid.map(vc.approach)
vd['airway'] = vd.caseid.map(vc.airway)
vd['tiva'] = np.where(vd.caseid.isin(tiva_ids), 'Propofol TCI (TIVA)', 'No propofol infusion')
vd['bmi_t'] = pd.cut(vd.bmi, [0, 21.5, 24.5, 100], labels=['<21.5', '21.5-24.5', '>24.5'])
st = []
for v in ['nibp_map', 'hr', 'etco2']:
    mods = fit_both(ref, v)
    hit = centile_of(mods, vd, v)
    d = vd.loc[hit.index]
    p50 = pd.concat([predict(mods[s], d[d.sex == s].age.values)[0.5].set_axis(d[d.sex == s].index) for s in mods]).loc[d.index]
    res = d[v] - p50
    for col, lab in (('approach', 'Surgical approach'), ('tiva', 'Anaesthetic'), ('bmi_t', 'BMI, kg/m2'), ('airway', 'Airway')):
        for lvl, g in d.groupby(col, observed=True):
            if len(g) < 30: continue
            st.append({'var': v, 'stratum': lab, 'level': str(lvl), 'n': len(g),
                       'below_P50': 100 * float(hit.loc[g.index, 0.5].mean()), 'median_shift': float(np.median(res[g.index]))})
pd.DataFrame(st).to_csv(f'{P}/out/validation_strata2.csv', index=False, float_format='%.4f')

# ---- E：MOVER 共變項
con = duckdb.connect()
ppf = con.execute(f"""select distinct LOG_ID from '{P}/out/mover_propofol_intraop.parquet'
                      where regexp_matches(upper(MEDICATION_NM),'INFUSION|/100 ?ML|/50 ?ML') and MAR_ACTION_NM in ('New Bag','Rate Change','Restarted')""").df()
ref['tiva'] = ref.LOG_ID.isin(set(ppf.LOG_ID)).astype(int)
ref['lap'] = ref.proc.fillna('').str.upper().str.contains('LAPAROSCOP|ROBOT').astype(int)
ref['bmi5'] = (ref.bmi - 25) / 5
cov = []
for v in ['nibp_map', 'hr', 'etco2']:
    d = ref[['age', 'sex', 'bmi5', 'lap', 'tiva', v]].dropna()
    for q in (0.03, 0.5, 0.97):
        m = smf.quantreg(f"{v} ~ cr(age, df=4) + C(sex) + bmi5 + lap + tiva", d).fit(q=q, max_iter=5000)
        ci = m.conf_int()
        for term, lab in (('bmi5', 'BMI, per 5 kg/m2'), ('lap', 'Laparoscopic or robotic'), ('tiva', 'Propofol infusion')):
            cov.append({'var': v, 'centile': int(round(q * 100)), 'term': lab, 'coef': m.params[term],
                        'lo': ci.loc[term, 0], 'hi': ci.loc[term, 1], 'n': len(d)})
covdf = pd.DataFrame(cov)
covdf.to_csv(f'{P}/out/covariates_qr.csv', index=False, float_format='%.5f')
json.dump({'n_ref': len(ref), 'pct_tiva': 100 * ref.tiva.mean(), 'pct_lap': 100 * ref.lap.mean(),
           'bmi_missing': int(ref.bmi.isna().sum())}, open(f'{P}/out/covariates_n.json', 'w'), indent=1)

# ---- F
rows = []
for name, d in (('All ASA I-II', ref), ('No vasoactive drug', nodrug), ('No vasoactive drug, anaesthesia >=120 min', nodrug[nodrug.dur >= 120])):
    for s, m in fit_both(d, 'etco2', [0.03, 0.5]).items():
        pr = predict(m, [30, 50, 70])
        rows.append({'population': name, 'sex': s, 'n': int((d.sex == s).sum()),
                     **{f'P3_{a}': pr.at[a, 0.03] for a in (30, 50, 70)}})
pd.DataFrame(rows).to_csv(f'{P}/out/etco2_nodrug_check.csv', index=False, float_format='%.1f')

pd.set_option('display.width', 220)
print(json.load(open(f'{P}/out/crossing.json')))
print(pd.DataFrame(rec).round(1).to_string())
print(pd.DataFrame(st).round(1).to_string())
print(covdf.round(2).to_string())
print(pd.DataFrame(rows).round(1))
