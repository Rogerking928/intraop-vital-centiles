"""共用：按性別的年齡平滑百分位（分位數迴歸＋自然三次樣條，df=4），預測時做 rearrangement 防交叉。"""
import numpy as np, pandas as pd, patsy, warnings
from statsmodels.regression.quantile_regression import QuantReg
from threadpoolctl import threadpool_limits

# 分位數迴歸在整數資料（EtCO2、取整的血壓）上解不唯一，多執行緒 BLAS 的加總順序會讓 IRLS 停在不同的解，
# 同一支程式重跑差到 0.1。鎖單執行緒，結果才可逐位重現（2026-10-01 實測兩次相同）。
threadpool_limits(1)

Q = [0.03, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97]
VARS = ['nibp_map', 'nibp_sbp', 'nibp_dbp', 'hr', 'etco2', 'temp_c', 'art_map']
LABEL = {'nibp_map': 'MAP, non-invasive (mmHg)', 'nibp_sbp': 'SBP, non-invasive (mmHg)',
         'nibp_dbp': 'DBP, non-invasive (mmHg)', 'hr': 'Heart rate (beats/min)', 'etco2': 'End-tidal CO2 (mmHg)',
         'temp_c': 'Temperature (°C)', 'art_map': 'MAP, arterial line (mmHg)', 'spo2': 'SpO2 (%)'}
DF = 4
AGE_KNOTS = dict(lower_bound=18, upper_bound=90)


def fit(d, var, qs=Q):
    """d: 一個性別的資料（age, var）。回傳 {q: (design_info, params)}。"""
    d = d[['age', var]].dropna()
    X = patsy.dmatrix(f"cr(age, df={DF}, lower_bound=18, upper_bound=90)", d, return_type='dataframe')
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for q in qs:
            out[q] = QuantReg(d[var].values, X.values).fit(q=q, max_iter=5000).params
    return X.design_info, out


def predict(model, ages):
    info, pars = model
    X = patsy.build_design_matrices([info], {'age': np.asarray(ages, float)})[0]
    qs = sorted(pars)
    P = np.column_stack([np.asarray(X) @ pars[q] for q in qs])
    P = np.sort(P, axis=1)                      # rearrangement（Chernozhukov 2010）
    return pd.DataFrame(P, columns=qs, index=np.asarray(ages))


def fit_both(d, var, qs=Q):
    return {s: fit(d[d.sex == s], var, qs) for s in ('Female', 'Male')}


def centile_of(models, d, var):
    """每個人的值落在自己年齡×性別曲線的哪個分位以下（回傳 DataFrame：每個 q 一欄 True/False）。"""
    rows = []
    for s, m in models.items():
        g = d[(d.sex == s)][['age', var]].dropna()
        if len(g) == 0:
            continue
        pr = predict(m, g.age.values)
        pr.index = g.index
        rows.append(pd.DataFrame({q: g[var].values < pr[q].values for q in pr.columns}, index=g.index))
    return pd.concat(rows)
