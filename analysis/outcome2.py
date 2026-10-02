"""Outcome analysis, second version (in response to review): the same quantity -- each patient's lowest 5-min sustained
MAP -- expressed on three scales, to see which discriminates outcomes better:
   absolute (mmHg), relative to pre-operative baseline (%), and age x sex reference centile (reference population's
   lowest sustained 5-min distribution).
Also: minutes below 65 including induction, in-hospital death, and the subgroup of women under 50.
Outcomes: postoperative AKI (KDIGO creatinine, same population as outcome.py); in-hospital death (discharge disposition
Expired, all analysable patients).
Pre-operative baseline: median PRE-OP MAP ('MAP (mmHg)', 'NIBP - MAP', or computed from 'BP'/'NIBP' systolic/diastolic)
within 6 h before anaesthesia start.
Each model: covariates (age spline, sex, ASA, anaesthesia duration, admission type; plus pre-operative Cr for AKI) + one
exposure (natural cubic spline, df=3).
Compares AIC and AUC; AUC difference vs absolute mmHg from 200 bootstrap resamples. Compared only in patients for whom
all three scales can be computed, so the comparison is fair.
Outputs out/outcome2_models.csv, out/outcome2_flags.csv, out/outcome2.json
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
disp = con.execute(f"select LOG_ID, max(DISCH_DISP) disp from read_csv_auto('{MX}/patient_information.csv', all_varchar=true) group by 1").df()
mv['death'] = mv.LOG_ID.map(disp.set_index('LOG_ID').disp).eq('Expired').astype(int)
con.register('pi', mv[['LOG_ID', 'an0', 'an1']])

# ---- Pre-operative baseline MAP
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

# ---- Reference centiles (lowest sustained 5 min)
ref = mv[mv.ref]
qs = [round(x, 3) for x in np.arange(0.005, 0.9951, 0.005)]
dense = fit_both(ref, 'maint_sust5', qs)
for s, m in dense.items():
    k = mv.sex == s
    pr = predict(m, mv.loc[k, 'age'].values).values
    mv.loc[k, 'sust5_centile'] = [100 * np.interp(v, row, qs, left=0.0, right=1.0) if np.isfinite(v) else np.nan
                                  for v, row in zip(mv.loc[k, 'maint_sust5'].values, pr)]
mv['sust5_pct_base'] = 100 * mv.maint_sust5 / mv.base_map

# ---- AKI population (same as outcome.py)
crl = con.execute(f"""
select l.LOG_ID, try_strptime(l."Collection Datetime", '%Y-%m-%d %H:%M:%S') t, try_cast(l."Observation Value" as double) cr,
       pi.an0, pi.an1
from read_csv_auto('{MX}/patient_labs.csv', all_varchar=true) l join pi using (LOG_ID)
where l."Lab Code" in ('2160-0','38483-4')""").df()
crl = crl[crl.cr.between(0.1, 20) & crl.t.notna()]
# When two values share a timestamp, take the higher one, for reproducibility (duckdb row order is not fixed)
p0 = crl[(crl.t < crl.an0) & (crl.t >= crl.an0 - pd.Timedelta(days=30))].sort_values(['LOG_ID', 't', 'cr'], kind='mergesort').groupby('LOG_ID').cr.last()
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

EXPO = {'Lowest sustained MAP, mmHg': ' + cr(maint_sust5, df=3, constraints="center")',
        'Lowest sustained MAP, % of pre-operative baseline': ' + cr(sust5_pct_base, df=3, constraints="center")',
        'Lowest sustained MAP, age-sex reference centile': ' + cr(sust5_centile, df=3, constraints="center")',
        'Minutes below 65 mmHg, maintenance': ' + l_maint65',
        'Minutes below 65 mmHg, whole anaesthesia incl. induction': ' + l_full65',
        'None (covariates only)': ''}


def fitm(f, d, y):
    m = smf.logit(f, d).fit(disp=0, maxiter=300)
    return m, roc_auc_score(d[y], m.predict(d))


EXPO_LIN = {'Lowest sustained MAP, mmHg': ' + maint_sust5',
            'Lowest sustained MAP, % of pre-operative baseline': ' + sust5_pct_base',
            'Lowest sustained MAP, age-sex reference centile': ' + sust5_centile',
            'Minutes below 65 mmHg, maintenance': ' + l_maint65',
            'Minutes below 65 mmHg, whole anaesthesia incl. induction': ' + l_full65',
            'None (covariates only)': ''}


def compare(d, y, base, label, expo=None):
    global EXPO
    keep = EXPO
    if expo is not None: EXPO = expo
    out, bs = [], {k: [] for k in EXPO}
    refk = 'Lowest sustained MAP, mmHg'
    for k, e in EXPO.items():
        m, auc = fitm(base + e, d, y)
        assert m.mle_retvals['converged'], f"{label} / {k}: not converged"
        out.append({'outcome': label, 'exposure': k, 'n': len(d), 'events': int(d[y].sum()), 'aic': m.aic, 'auc': auc,
                    'df': int(m.df_model), 'n_params': len(m.params) - 1})
    rng = np.random.default_rng(5)
    for b in range(int(os.environ.get('NBOOT', 200))):
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
    EXPO = keep
    return r


need = ['maint_sust5', 'sust5_pct_base', 'sust5_centile']
flow = {'known_asa': len(mv), 'with_baseline_map': int(mv.base_map.notna().sum())}
aki = mv[mv.cr0.notna() & mv.cr7.notna() & (mv.cr0 < 4.0)].dropna(subset=need)
dth = mv.dropna(subset=need).copy()
flow.update({'aki_cohort': len(aki), 'aki_events': int(aki.aki.sum()), 'death_cohort': len(dth),
             'deaths': int(dth.death.sum())})
B_AKI = "aki ~ cr(age, df=4, constraints='center') + C(sex) + C(asa_c) + cr0 + log_dur + inpatient"
# Splines are always centred (constraints='center'): uncentred cr() is collinear with the intercept, so df_model
# varies with numerical tolerance and the AIC is wrong.
# Death: no deaths in ASA I (only 2 in ASA II), so the ASA I coefficient diverges and the Hessian is singular ->
# the death model merges ASA I-II into one group
dth['asa_d'] = np.where(dth.asa <= 2, '1-2', dth.asa_c)
flow['deaths_asa12'] = int(dth.death[dth.asa <= 2].sum()); flow['n_asa12_death_cohort'] = int((dth.asa <= 2).sum())
B_DTH = "death ~ cr(age, df=4, constraints='center') + C(sex) + C(asa_d) + log_dur + inpatient"
res = pd.concat([compare(aki, 'aki', B_AKI, 'Acute kidney injury'),
                 compare(dth, 'death', B_DTH, 'In-hospital death')])
# Women under 50: only a dozen or so outpatients and no AKI among them (complete separation by admission type), so
# inpatient is left out of the model. Only AKI is reported for this subgroup.
yw = aki[(aki.sex == 'Female') & (aki.age < 50)].copy()
flow.update({'young_women_aki_cohort': len(yw), 'young_women_aki': int(yw.aki.sum()),
             'young_women_outpatient': int((yw.inpatient == 0).sum()), 'young_women_outpatient_aki': int(yw.aki[yw.inpatient == 0].sum())})
res = pd.concat([res, compare(yw, 'aki', "aki ~ cr(age, df=3, constraints='center') + C(asa_c) + cr0 + log_dur",
                              'Acute kidney injury, women under 50')])
# Only about 160 deaths: a parsimonious model (linear age, sex, ASA III or above, anaesthesia duration; linear exposure)
# as a sensitivity analysis
dth['asa3'] = (dth.asa >= 3).astype(int)
res = pd.concat([res, compare(dth, 'death', "death ~ age + C(sex) + asa3 + log_dur", 'In-hospital death, parsimonious model',
                              EXPO_LIN)])
# (revision 2026-10-01: % baseline also carries information from the pre-operative MAP itself. Add pre-operative MAP as
# a covariate and check whether each scale still adds anything.)
res = pd.concat([res,
                 compare(dth, 'death', B_DTH + " + cr(base_map, df=3, constraints='center')",
                         'In-hospital death, adjusted for pre-operative MAP'),
                 compare(dth, 'death', "death ~ age + C(sex) + asa3 + log_dur + base_map",
                         'In-hospital death, parsimonious model, adjusted for pre-operative MAP', EXPO_LIN),
                 compare(aki, 'aki', B_AKI + " + cr(base_map, df=3, constraints='center')",
                         'Acute kidney injury, adjusted for pre-operative MAP')])
# Direction of association between pre-operative MAP itself and death (review: whether low pre-operative pressure
# indicates sicker patients)
mb = smf.logit("death ~ age + C(sex) + asa3 + log_dur + I(base_map / 10)", dth).fit(disp=0)
k = 'I(base_map / 10)'
flow['death_or_per10_base_map'] = float(np.exp(mb.params[k]))
flow['death_or_per10_base_map_ci'] = [float(x) for x in np.exp(mb.conf_int().loc[k])]
dth['base_q'] = pd.qcut(dth.base_map, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
bq = dth.groupby('base_q', observed=True).agg(n=('death', 'size'), deaths=('death', 'sum'), lo=('base_map', 'min'),
                                              hi=('base_map', 'max'))
bq['pct'] = 100 * bq.deaths / bq.n
bq.to_csv(f'{P}/out/outcome2_baseline_quartiles.csv', float_format='%.12g')
res.to_csv(f'{P}/out/outcome2_models.csv', index=False, float_format='%.12g')

# Proportion flagged as hypotensive and their AKI incidence: absolute <65, relative <80% of baseline, centile <P10
# (all on lowest sustained 5 min)
fl = []
for name, g in (('All', aki), ('Women under 50', yw), ('Men 70 and over', aki[(aki.sex == 'Male') & (aki.age >= 70)])):
    for lab, mask in (('Below 65 mmHg', g.maint_sust5 < 65), ('Below 80% of baseline', g.sust5_pct_base < 80),
                      ('Below reference P10', g.sust5_centile < 10)):
        fl.append({'group': name, 'definition': lab, 'n': len(g), 'n_flagged': int(mask.sum()),
                   'aki_events_flagged': int(g.aki[mask].sum()), 'flagged_pct': 100 * mask.mean(),
                   'aki_flagged_pct': 100 * g.aki[mask].mean() if mask.any() else np.nan,
                   'aki_not_flagged_pct': 100 * g.aki[~mask].mean() if (~mask).any() else np.nan})
pd.DataFrame(fl).to_csv(f'{P}/out/outcome2_flags.csv', index=False, float_format='%.12g')
flow['baseline_map_median'] = float(aki.base_map.median())
flow['aki_asa_ge3_pct'] = 100 * float((aki.asa >= 3).mean())
flow['aki_pct'] = 100 * float(aki.aki.mean())
json.dump(flow, open(f'{P}/out/outcome2.json', 'w'), indent=1)
pd.set_option('display.width', 250)
print(json.dumps(flow, indent=1)); print(res.round(4).to_string()); print(pd.DataFrame(fl).round(1).to_string())
