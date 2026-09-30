"""預後分析第二版（回應審閱）：同一個量——每人持續 5 分鐘的最低 MAP——用三種尺度表示，誰對結局的區辨力較好：
   絕對值（mmHg）、相對術前基線（%）、年齡×性別參考百分位（參考族群的 lowest sustained 5-min 分布）。
另加：含誘導期的低於 65 分鐘數、住院死亡、50 歲以下女性子群。
結局：術後 AKI（KDIGO 肌酸酐，同 outcome.py 的族群）；住院死亡（出院去向 Expired，全部可分析者）。
術前基線：PRE-OP 的 MAP（'MAP (mmHg)'、'NIBP - MAP'，或由 'BP'/'NIBP' 的收縮／舒張壓算）在麻醉開始前 6 小時內的中位數。
每個模型：共變項（年齡樣條、性別、ASA、麻醉時長、住院類別；AKI 另加術前 Cr）＋一個暴露（自然三次樣條 df=3）。
比較 AIC、AUC；bootstrap 200 次的 AUC 差（對絕對 mmHg）。只在三個尺度都算得出來的人上比，才公平。
輸出 out/outcome2_models.csv、out/outcome2_flags.csv、out/outcome2.json
"""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, sys, json, duckdb, numpy as np, pandas as pd, warnings
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfit import fit_both, predict
warnings.simplefilter('ignore')

P = _PROJ
MX = _os.path.join(_MOVER, 'EPIC_EMR/EMR')
con = duckdb.connect()
con.execute("set memory_limit='4GB'; set threads=6")
mv = pd.read_parquet(f'{P}/out/cohort_mover.parquet').merge(pd.read_parquet(f'{P}/out/readings_mover.parquet'), on='LOG_ID')
mv = mv[mv.asa.notna()].copy()
disp = con.execute(f"select LOG_ID, any_value(DISCH_DISP) disp from read_csv_auto('{MX}/patient_information.csv', all_varchar=true) group by 1").df()
mv['death'] = mv.LOG_ID.map(disp.set_index('LOG_ID').disp).eq('Expired').astype(int)
con.register('pi', mv[['LOG_ID', 'an0', 'an1']])

# ---- 術前基線 MAP
pre = con.execute(f"""
select p.LOG_ID, p.t, p.item, p.v from '{P}/out/mover_preop/*.parquet' p join pi using (LOG_ID)
where p.rt='PRE-OP' and p.t < pi.an0 and p.t >= pi.an0 - interval 6 hour""").df()


def to_map(r):
    try:
        if r.item in ('MAP (mmHg)', 'NIBP - MAP'):
            return float(r.v)
        s, d = str(r.v).split('/')[:2]
        s, d = float(s), float(d)
        return d + (s - d) / 3 if s > d else np.nan
    except Exception:
        return np.nan


pre['mp'] = pre.apply(to_map, axis=1)
pre = pre[pre.mp.between(40, 180)]
mv['base_map'] = mv.LOG_ID.map(pre.groupby('LOG_ID').mp.median())

# ---- 參考百分位（lowest sustained 5 min）
ref = mv[mv.ref]
qs = [round(x, 3) for x in np.arange(0.005, 0.9951, 0.005)]
dense = fit_both(ref, 'maint_sust5', qs)
for s, m in dense.items():
    k = mv.sex == s
    pr = predict(m, mv.loc[k, 'age'].values).values
    mv.loc[k, 'sust5_centile'] = [100 * np.interp(v, row, qs, left=0.0, right=1.0) if np.isfinite(v) else np.nan
                                  for v, row in zip(mv.loc[k, 'maint_sust5'].values, pr)]
mv['sust5_pct_base'] = 100 * mv.maint_sust5 / mv.base_map

# ---- AKI 族群（同 outcome.py）
crl = con.execute(f"""
select l.LOG_ID, try_strptime(l."Collection Datetime", '%Y-%m-%d %H:%M:%S') t, try_cast(l."Observation Value" as double) cr,
       pi.an0, pi.an1
from read_csv_auto('{MX}/patient_labs.csv', all_varchar=true) l join pi using (LOG_ID)
where l."Lab Code" in ('2160-0','38483-4')""").df()
crl = crl[crl.cr.between(0.1, 20) & crl.t.notna()]
p0 = crl[(crl.t < crl.an0) & (crl.t >= crl.an0 - pd.Timedelta(days=30))].sort_values('t').groupby('LOG_ID').cr.last()
post = crl[crl.t >= crl.an1]
p48 = post[post.t < post.an1 + pd.Timedelta(hours=48)].groupby('LOG_ID').cr.max()
p7 = post[post.t < post.an1 + pd.Timedelta(days=7)].groupby('LOG_ID').cr.max()
mv['cr0'], mv['cr48'], mv['cr7'] = mv.LOG_ID.map(p0), mv.LOG_ID.map(p48), mv.LOG_ID.map(p7)
mv['aki'] = (((mv.cr48 - mv.cr0) >= 0.3) | (mv.cr7 >= 1.5 * mv.cr0)).astype(int)
mv['inpatient'] = (mv.cls != 'Hospital Outpatient Surgery').astype(int)
mv['asa_c'] = mv.asa.clip(upper=4).astype(int).astype(str)
mv['log_dur'] = np.log(mv.dur)
mv['l_full65'] = np.log1p(mv.full_min_lt65)
mv['l_maint65'] = np.log1p(mv.maint_min_lt65)

EXPO = {'Lowest sustained MAP, mmHg': ' + cr(maint_sust5, df=3)',
        'Lowest sustained MAP, % of pre-operative baseline': ' + cr(sust5_pct_base, df=3)',
        'Lowest sustained MAP, age-sex reference centile': ' + cr(sust5_centile, df=3)',
        'Minutes below 65 mmHg, maintenance': ' + l_maint65',
        'Minutes below 65 mmHg, whole anaesthesia incl. induction': ' + l_full65',
        'None (covariates only)': ''}


def fitm(f, d, y):
    m = smf.logit(f, d).fit(disp=0, maxiter=300)
    return m, roc_auc_score(d[y], m.predict(d))


def compare(d, y, base, label):
    out, bs = [], {k: [] for k in EXPO}
    refk = 'Lowest sustained MAP, mmHg'
    for k, e in EXPO.items():
        m, auc = fitm(base + e, d, y)
        out.append({'outcome': label, 'exposure': k, 'n': len(d), 'events': int(d[y].sum()), 'aic': m.aic, 'auc': auc})
    rng = np.random.default_rng(5)
    for b in range(200):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        a = {}
        for k, e in EXPO.items():
            try: a[k] = fitm(base + e, s, y)[1]
            except Exception: a[k] = np.nan
        for k in EXPO: bs[k].append(a[k] - a[refk])
    r = pd.DataFrame(out)
    r['daic_vs_mmHg'] = r.aic - r.set_index('exposure').at[refk, 'aic']
    r['dauc_vs_mmHg'] = r.auc - r.set_index('exposure').at[refk, 'auc']
    r['dauc_lo'] = [np.nanpercentile(bs[k], 2.5) for k in r.exposure]
    r['dauc_hi'] = [np.nanpercentile(bs[k], 97.5) for k in r.exposure]
    return r


need = ['maint_sust5', 'sust5_pct_base', 'sust5_centile']
flow = {'known_asa': len(mv), 'with_baseline_map': int(mv.base_map.notna().sum())}
aki = mv[mv.cr0.notna() & mv.cr7.notna() & (mv.cr0 < 4.0)].dropna(subset=need)
dth = mv.dropna(subset=need)
flow.update({'aki_cohort': len(aki), 'aki_events': int(aki.aki.sum()), 'death_cohort': len(dth),
             'deaths': int(dth.death.sum())})
B_AKI = "aki ~ cr(age, df=4) + C(sex) + C(asa_c) + cr0 + log_dur + inpatient"
B_DTH = "death ~ cr(age, df=4) + C(sex) + C(asa_c) + log_dur + inpatient"
res = pd.concat([compare(aki, 'aki', B_AKI, 'Acute kidney injury'),
                 compare(dth, 'death', B_DTH, 'In-hospital death')])
# 50 歲以下女性：AKI 事件太少時改用住院死亡以外的，只報 AKI
yw = aki[(aki.sex == 'Female') & (aki.age < 50)]
flow.update({'young_women_aki_cohort': len(yw), 'young_women_aki': int(yw.aki.sum())})
res = pd.concat([res, compare(yw, 'aki', "aki ~ cr(age, df=3) + C(asa_c) + cr0 + log_dur + inpatient",
                              'Acute kidney injury, women under 50')])
res.to_csv(f'{P}/out/outcome2_models.csv', index=False, float_format='%.4f')

# 被標記為低血壓的比例與其 AKI 發生率：絕對 <65、相對 <80% 基線、百分位 <P10（都用 lowest sustained 5 min）
fl = []
for name, g in (('All', aki), ('Women under 50', yw), ('Men 70 and over', aki[(aki.sex == 'Male') & (aki.age >= 70)])):
    for lab, mask in (('Below 65 mmHg', g.maint_sust5 < 65), ('Below 80% of baseline', g.sust5_pct_base < 80),
                      ('Below reference P10', g.sust5_centile < 10)):
        fl.append({'group': name, 'definition': lab, 'n': len(g), 'flagged_pct': 100 * mask.mean(),
                   'aki_flagged_pct': 100 * g.aki[mask].mean() if mask.any() else np.nan,
                   'aki_not_flagged_pct': 100 * g.aki[~mask].mean() if (~mask).any() else np.nan})
pd.DataFrame(fl).to_csv(f'{P}/out/outcome2_flags.csv', index=False, float_format='%.2f')
flow['baseline_map_median'] = float(aki.base_map.median())
json.dump(flow, open(f'{P}/out/outcome2.json', 'w'), indent=1)
pd.set_option('display.width', 250)
print(json.dumps(flow, indent=1)); print(res.round(4).to_string()); print(pd.DataFrame(fl).round(1).to_string())
