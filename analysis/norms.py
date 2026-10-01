"""MOVER 參考族群（ASA I–II）的年齡×性別百分位曲線。

主分析：每人維持期中位數 ~ 性別別分位數迴歸（自然三次樣條 df=4）。
輸出
  out/norms_table.csv          18–90 每一歲、每性別、每變數、七個分位（Supplementary 參考表）
  out/norms_ci.csv             每 10 歲的 P3/P50/P97 與 bootstrap 95% CI（200 次，以病人重抽）
  out/norms_cv.csv             5 折交叉驗證：留出的人落在各分位以下的比例（依年齡段）
  out/norms_sensitivity.csv    換族群重估（全部 ASA、排除升壓劑輸注、只門診手術）P3/P50/P97
  out/threshold.csv            MAP 55／60／65 mmHg 在各年齡段×性別落在第幾百分位＋實際低於 65 的比例
  out/spo2_desc.csv            SpO2 描述（天花板效應，不做曲線）
"""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, sys, json, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfit import Q, fit_both, predict, centile_of, fit

P = _PROJ
V = ['nibp_map', 'nibp_sbp', 'nibp_dbp', 'hr', 'etco2', 'temp_c']
AGES = np.arange(18, 91)
DEC = np.array([20, 30, 40, 50, 60, 70, 80, 90])
mv = pd.read_parquet(f'{P}/out/cohort_mover.parquet')
ref = mv[mv.ref].reset_index(drop=True)
BANDS = [18, 30, 40, 50, 60, 70, 80, 91]
BLAB = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-90']


def table(models, var):
    rows = []
    for s, m in models.items():
        pr = predict(m, AGES)
        for a in AGES:
            rows.append({'var': var, 'sex': s, 'age': int(a), **{f'P{int(round(q*100))}': pr.at[a, q] for q in pr.columns}})
    return rows


def boot_one(seed):
    rng = np.random.default_rng(seed)
    out = []
    for s in ('Female', 'Male'):
        g = ref[ref.sex == s]
        b = g.iloc[rng.integers(0, len(g), len(g))]
        for v in V:
            pr = predict(fit(b, v, [0.03, 0.5, 0.97]), DEC)
            for a in DEC:
                out.append((v, s, int(a), pr.at[a, 0.03], pr.at[a, 0.5], pr.at[a, 0.97]))
    return out


if __name__ == '__main__':
    models = {v: fit_both(ref, v) for v in V}
    tab = pd.DataFrame(sum((table(models[v], v) for v in V), []))
    tab.to_csv(f'{P}/out/norms_table.csv', index=False, float_format='%.12g')

    # bootstrap CI
    with Pool(6) as p:
        res = sum(p.map(boot_one, range(200)), [])
    b = pd.DataFrame(res, columns=['var', 'sex', 'age', 'P3', 'P50', 'P97'])
    ci = b.groupby(['var', 'sex', 'age']).quantile([0.025, 0.975]).unstack()
    ci.columns = [f'{c}_{"lo" if q < 0.5 else "hi"}' for c, q in ci.columns]
    pt = tab[tab.age.isin(DEC)].set_index(['var', 'sex', 'age'])[['P3', 'P50', 'P97']]
    ci = pt.join(ci)[['P3', 'P3_lo', 'P3_hi', 'P50', 'P50_lo', 'P50_hi', 'P97', 'P97_lo', 'P97_hi']]
    ci.to_csv(f'{P}/out/norms_ci.csv', float_format='%.12g')

    # 5 折交叉驗證
    rng = np.random.default_rng(1)
    fold = rng.integers(0, 5, len(ref))
    ref['band'] = pd.cut(ref.age, BANDS, right=False, labels=BLAB)
    cv = []
    for v in V:
        hits = []
        for k in range(5):
            mdl = fit_both(ref[fold != k], v)
            hits.append(centile_of(mdl, ref[fold == k], v))
        h = pd.concat(hits).join(ref[['band', 'sex']])
        allrow = {'var': v, 'band': 'All', 'n': len(h), **{f'below_P{int(round(q*100))}': h[q].mean() for q in Q}}
        cv.append(allrow)
        for bnd, g in h.groupby('band', observed=True):
            cv.append({'var': v, 'band': bnd, 'n': len(g), **{f'below_P{int(round(q*100))}': g[q].mean() for q in Q}})
    pd.DataFrame(cv).to_csv(f'{P}/out/norms_cv.csv', index=False, float_format='%.12g')

    # 敏感度：換族群
    # MOVER 的日期逐病人隨機位移（Samad 2023），所以不做分年份的敏感度分析
    pops = {'Primary (ASA I-II)': mv.ref,
            'All ASA classes': mv.asa.notna(),
            'ASA I-II, no vasoactive infusion': mv.ref & (mv.vaso_inf == 0),
            'ASA I-II, no vasoactive drug': mv.ref & (mv.vaso_inf == 0) & (mv.vaso_bolus == 0),
            'ASA I-II, outpatient surgery': mv.ref & (mv.cls == 'Hospital Outpatient Surgery'),
            'ASA I-II, no vasoactive drug, anaesthesia >=120 min': mv.ref & (mv.vaso_inf == 0) & (mv.vaso_bolus == 0) & (mv.dur >= 120)}
    sens = []
    for name, mask in pops.items():
        d = mv[mask]
        for v in V:
            for s, m in fit_both(d, v, [0.03, 0.5, 0.97]).items():
                pr = predict(m, DEC)
                for a in DEC:
                    sens.append({'population': name, 'n': int(len(d)), 'var': v, 'sex': s, 'age': int(a),
                                 'P3': pr.at[a, 0.03], 'P50': pr.at[a, 0.5], 'P97': pr.at[a, 0.97]})
    pd.DataFrame(sens).to_csv(f'{P}/out/norms_sensitivity.csv', index=False, float_format='%.12g')

    # 固定門檻（55／60／65 mmHg）在各年齡段中點、各性別落在第幾百分位（密集分位網格反推）
    qs = [round(x, 3) for x in np.arange(0.005, 0.5001, 0.005)]
    dense = fit_both(ref, 'nibp_map', qs)
    MID = {'18-29': 24, '30-39': 35, '40-49': 45, '50-59': 55, '60-69': 65, '70-79': 75, '80-90': 85}
    thr = []
    for s_, m in dense.items():
        pr = predict(m, list(MID.values()))
        for b, a in MID.items():
            row = pr.loc[a].values
            r = {'band': b, 'sex': s_, 'age_mid': a}
            for T in (55, 60, 65):
                r[f'centile_of_{T}'] = 100 * np.interp(T, row, np.array(pr.columns, float), left=0.0, right=0.5)
            thr.append(r)
    thr = pd.DataFrame(thr).set_index(['band', 'sex'])
    g = ref.groupby(['band', 'sex'], observed=True)
    emp = pd.DataFrame({'n': g.size(),
                        'pct_median_lt65': g.nibp_map.apply(lambda x: 100 * (x < 65).mean()),
                        'pct_min_lt65': g.nibp_min_map.apply(lambda x: 100 * (x < 65).mean()),
                        'pct_time_lt65': g.nibp_frac_lt65.apply(lambda x: 100 * x.mean())})
    thr.join(emp).to_csv(f'{P}/out/threshold.csv', float_format='%.12g')
    tot = {'pct_min_lt65': 100 * (ref.nibp_min_map < 65).mean(), 'pct_median_lt65': 100 * (ref.nibp_map < 65).mean(),
           'pct_time_lt65': 100 * ref.nibp_frac_lt65.mean(), 'n': len(ref)}
    json.dump(tot, open(f'{P}/out/threshold_overall.json', 'w'), indent=1)

    sp = ref.groupby(['band', 'sex'], observed=True).spo2.describe(percentiles=[0.03, 0.1, 0.5])
    sp['pct_at_100'] = ref.groupby(['band', 'sex'], observed=True).spo2.apply(lambda x: 100 * (x >= 100).mean())
    sp.to_csv(f'{P}/out/spo2_desc.csv', float_format='%.12g')

    pd.set_option('display.width', 250)
    print(ci.round(1).to_string())
    print(pd.DataFrame(cv).query("band=='All'").round(3).to_string())
    print(thr.join(emp).round(1).to_string()); print(tot)
