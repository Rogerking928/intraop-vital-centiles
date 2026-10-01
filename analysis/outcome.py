"""術後急性腎損傷（AKI）：固定 65 mmHg vs 年齡×性別參考百分位當低血壓閾值。

族群：MOVER 可分析者（全部 ASA），有術前 30 天內肌酸酐、術後 7 天內至少一次肌酸酐，術前 Cr <4.0 mg/dL。
AKI（KDIGO 肌酸酐標準）：術後 48 小時內最高值 ≥ 術前 +0.3 mg/dL，或 7 天內最高值 ≥ 術前 ×1.5。
暴露（維持期 NIBP MAP，每筆讀值代表到下一筆的時間，上限 10 分鐘）：
  低於 65 mmHg 的分鐘數；低於自己年齡×性別參考 P3、P10 的分鐘數；
  另：病人的維持期 MAP 中位數，以 mmHg 表示 vs 以參考百分位表示。
模型：logistic，校正年齡（樣條）、性別、ASA、術前 Cr、麻醉時長、住院類別；
比較 AIC 與 AUC（加入暴露前後），bootstrap 200 次看 AUC 差的 CI。
輸出 out/outcome.json、out/outcome_models.csv、out/outcome_dose.csv
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

P = _PROJ
MX = _os.path.join(_MOVER, 'EPIC_EMR/EMR')
warnings.simplefilter('ignore')
con = duckdb.connect()
con.execute("set memory_limit='4GB'; set threads=6")

mv = pd.read_parquet(f'{P}/out/cohort_mover.parquet')
mv = mv[mv.asa.notna()].copy()
con.register('pi', mv[['LOG_ID', 'an0', 'an1']])

# ---- 肌酸酐（血清／全血） ----
crl = con.execute(f"""
select l.LOG_ID, try_strptime(l."Collection Datetime", '%Y-%m-%d %H:%M:%S') t, try_cast(l."Observation Value" as double) cr,
       pi.an0, pi.an1
from read_csv_auto('{MX}/patient_labs.csv', all_varchar=true) l join pi using (LOG_ID)
where l."Lab Code" in ('2160-0','38483-4')""").df()
crl = crl[crl.cr.between(0.1, 20) & crl.t.notna()]
# 同一時間兩筆時取較高者，結果才可重現
pre = crl[(crl.t < crl.an0) & (crl.t >= crl.an0 - pd.Timedelta(days=30))].sort_values(['LOG_ID', 't', 'cr'], kind='mergesort').groupby('LOG_ID').cr.last()
post = crl[crl.t >= crl.an1]
p48 = post[post.t < post.an1 + pd.Timedelta(hours=48)].groupby('LOG_ID').cr.max()
p7 = post[post.t < post.an1 + pd.Timedelta(days=7)].groupby('LOG_ID').cr.max()
mv['cr0'] = mv.LOG_ID.map(pre); mv['cr48'] = mv.LOG_ID.map(p48); mv['cr7'] = mv.LOG_ID.map(p7)
flow = {'analysable_known_asa': len(mv), 'baseline_cr': int(mv.cr0.notna().sum())}
d = mv[mv.cr0.notna() & mv.cr7.notna()].copy()
flow['baseline_and_postop_cr'] = len(d)
d = d[d.cr0 < 4.0].copy()
flow['baseline_cr_lt4'] = len(d)
d['aki'] = (((d.cr48 - d.cr0) >= 0.3) | (d.cr7 >= 1.5 * d.cr0)).astype(int)
flow['aki'] = int(d.aki.sum())

# ---- 參考閾值：參考族群的 P3／P10（每人一個數） ----
ref = mv[mv.ref]
models = fit_both(ref, 'nibp_map', [0.03, 0.10, 0.5])
for s, m in models.items():
    k = d.sex == s
    pr = predict(m, d.loc[k, 'age'].values)
    d.loc[k, 'thr_p3'] = pr[0.03].values
    d.loc[k, 'thr_p10'] = pr[0.10].values
    # 病人自己 MAP 中位數的參考百分位（用 0.5% 格點反推）
qs = [round(x, 3) for x in np.arange(0.005, 0.9951, 0.005)]
dense = fit_both(ref, 'nibp_map', qs)
for s, m in dense.items():
    k = d.sex == s
    pr = predict(m, d.loc[k, 'age'].values).values
    x = d.loc[k, 'nibp_map'].values
    d.loc[k, 'map_centile'] = [100 * np.interp(v, row, qs, left=0.0, right=1.0) for v, row in zip(x, pr)]

# ---- 維持期 NIBP 讀值與低於閾值的分鐘數 ----
con.register('dd', d[['LOG_ID', 'an0', 'an1', 'thr_p3', 'thr_p10']])
exp = con.execute(f"""
with r as (
  select a.LOG_ID, a.t, try_cast(split_part(a.v,'/',1) as double) sbp, try_cast(split_part(a.v,'/',2) as double) dbp,
         try_cast(b.v as double) mp
  from '{P}/out/mover_intraop/*.parquet' a
  join '{P}/out/mover_intraop/*.parquet' b on a.LOG_ID=b.LOG_ID and a.t=b.t
  join dd on a.LOG_ID = dd.LOG_ID
  where a.FLO_NAME='Devices Testing Template' and b.FLO_NAME='Devices Testing Template'
    and a.item='NIBP' and b.item='NIBP - MAP'
    and a.t >= dd.an0 + interval 15 minute and a.t < dd.an1 - interval 15 minute),
ok as (select * from r where sbp < 250 and sbp - dbp > 5 and dbp <= mp and mp <= sbp and mp between 30 and 200 and dbp >= 10),
w as (select ok.*, least(coalesce(date_diff('second', t, lead(t) over (partition by LOG_ID order by t)) / 60.0, 5), 10) dur
      from ok)
select w.LOG_ID, sum(case when mp < 65 then dur else 0 end) min_lt65,
       sum(case when mp < dd.thr_p3 then dur else 0 end) min_ltp3,
       sum(case when mp < dd.thr_p10 then dur else 0 end) min_ltp10,
       sum(case when mp < 55 then dur else 0 end) min_lt55,
       sum(dur) min_total
from w join dd using (LOG_ID) group by 1""").df()
d = d.merge(exp, on='LOG_ID', how='inner')
flow['with_exposure'] = len(d)
d['inpatient'] = (d.cls != 'Hospital Outpatient Surgery').astype(int)
d['asa_c'] = d.asa.clip(upper=4).astype(int).astype(str)
d['log_dur'] = np.log(d.dur)
for c in ['min_lt65', 'min_ltp3', 'min_ltp10', 'min_lt55']:
    d[f'l_{c}'] = np.log1p(d[c])

BASE = "aki ~ cr(age, df=4, constraints='center') + C(sex) + C(asa_c) + cr0 + log_dur + inpatient"
EXPO = {'None (covariates only)': '',
        'Minutes below 65 mmHg': ' + l_min_lt65',
        'Minutes below 55 mmHg': ' + l_min_lt55',
        'Minutes below own reference P10': ' + l_min_ltp10',
        'Minutes below own reference P3': ' + l_min_ltp3',
        'Case median MAP, mmHg (spline)': ' + cr(nibp_map, df=4, constraints="center")',
        'Case median MAP, reference centile (spline)': ' + cr(map_centile, df=4, constraints="center")'}


def fitm(f, data):
    m = smf.logit(f, data).fit(disp=0, maxiter=200)
    return m, roc_auc_score(data.aki, m.predict(data))


rows = []
fits = {}
for name, e in EXPO.items():
    m, auc = fitm(BASE + e, d)
    fits[name] = m
    term = e.strip(' +')
    orr = (np.exp(m.params[term]), *np.exp(m.conf_int().loc[term])) if term.startswith('l_') else (np.nan,) * 3
    rows.append({'exposure': name, 'aic': m.aic, 'auc': auc, 'or_per_log_min': orr[0], 'or_lo': orr[1], 'or_hi': orr[2]})
res = pd.DataFrame(rows)
res['delta_aic_vs_65'] = res.aic - res.set_index('exposure').at['Minutes below 65 mmHg', 'aic']

# bootstrap：同一批人，每個暴露的 AUC 減去「低於 65」的 AUC
rng = np.random.default_rng(11)
bs = {k: [] for k in EXPO}
for b in range(200):
    s = d.iloc[rng.integers(0, len(d), len(d))]
    a = {}
    for name, e in EXPO.items():
        try:
            a[name] = fitm(BASE + e, s)[1]
        except Exception:
            a[name] = np.nan
    for k in EXPO:
        bs[k].append(a[k] - a['Minutes below 65 mmHg'])
res['dauc_vs_65'] = res.auc - res.set_index('exposure').at['Minutes below 65 mmHg', 'auc']
res['dauc_lo'] = [np.nanpercentile(bs[k], 2.5) for k in res.exposure]
res['dauc_hi'] = [np.nanpercentile(bs[k], 97.5) for k in res.exposure]
res.to_csv(f'{P}/out/outcome_models.csv', index=False, float_format='%.4f')

# 劑量反應：暴露分鐘數分組的 AKI 發生率（未校正）
dose = []
for c, lab in [('min_lt65', 'Below 65 mmHg'), ('min_ltp3', 'Below own P3'), ('min_ltp10', 'Below own P10')]:
    g = pd.cut(d[c], [-0.1, 0, 5, 15, 30, np.inf], labels=['0', '>0-5', '>5-15', '>15-30', '>30'])
    t = d.groupby(g, observed=True).aki.agg(['size', 'sum', 'mean']).reset_index()
    t['threshold'] = lab
    t.columns = ['minutes', 'n', 'aki', 'rate', 'threshold']
    dose.append(t)
dose = pd.concat(dose)
dose['rate'] *= 100
dose.to_csv(f'{P}/out/outcome_dose.csv', index=False, float_format='%.2f')

# 年齡層：同樣 0 分鐘 vs 有暴露，看 65 與 P10 誰在年輕人／老人各抓到誰
d['band'] = pd.cut(d.age, [18, 50, 70, 91], right=False, labels=['18-49', '50-69', '70-90'])
by = d.groupby('band', observed=True).apply(lambda g: pd.Series({
    'n': len(g), 'aki_pct': 100 * g.aki.mean(),
    'any_lt65_pct': 100 * (g.min_lt65 > 0).mean(), 'any_ltp10_pct': 100 * (g.min_ltp10 > 0).mean(),
    'thr_p10_median': g.thr_p10.median(), 'thr_p3_median': g.thr_p3.median()}))
by.to_csv(f'{P}/out/outcome_by_age.csv', float_format='%.2f')

out = {'flow': flow, 'aki_pct': 100 * d.aki.mean(), 'age_median': float(d.age.median()),
       'asa_ge3_pct': 100 * float((d.asa >= 3).mean()),
       'min_lt65_median_among_exposed': float(d.min_lt65[d.min_lt65 > 0].median()),
       'any_lt65_pct': 100 * float((d.min_lt65 > 0).mean())}
json.dump(out, open(f'{P}/out/outcome.json', 'w'), indent=1)
pd.set_option('display.width', 220)
print(json.dumps(out, indent=1)); print(res.round(4).to_string()); print(dose.to_string()); print(by.round(1))
