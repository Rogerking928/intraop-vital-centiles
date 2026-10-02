"""Shared: sex-specific age-smoothed centiles (quantile regression on a natural cubic regression spline of age, cr(df=4):
four basis functions that, with the intercept, give three degrees of freedom for age); predictions are
rearranged to prevent crossing."""
import numpy as np, pandas as pd, patsy, warnings
from statsmodels.regression.quantile_regression import QuantReg
from threadpoolctl import threadpool_limits

# Quantile regression on integer data (EtCO2, rounded blood pressure) has non-unique solutions; multithreaded BLAS summation
# order makes IRLS stop at different solutions, so reruns differed by up to 0.1. Pinning to one thread makes results
# bit-for-bit reproducible (checked 2026-10-01: two runs identical).
threadpool_limits(1)

Q = [0.03, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97]
VARS = ['nibp_map', 'nibp_sbp', 'nibp_dbp', 'hr', 'etco2', 'temp_c', 'art_map']
LABEL = {'nibp_map': 'MAP, non-invasive (mmHg)', 'nibp_sbp': 'SBP, non-invasive (mmHg)',
         'nibp_dbp': 'DBP, non-invasive (mmHg)', 'hr': 'Heart rate (beats/min)', 'etco2': 'End-tidal CO2 (mmHg)',
         'temp_c': 'Temperature (°C)', 'art_map': 'MAP, arterial line (mmHg)', 'spo2': 'SpO2 (%)'}
DF = 4
AGE_KNOTS = dict(lower_bound=18, upper_bound=90)


def fit(d, var, qs=Q):
    """d: data for one sex (age, var). Returns {q: (design_info, params)}."""
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
    P = np.sort(P, axis=1)                      # rearrangement (Chernozhukov 2010)
    return pd.DataFrame(P, columns=qs, index=np.asarray(ages))


def fit_both(d, var, qs=Q):
    return {s: fit(d[d.sex == s], var, qs) for s in ('Female', 'Male')}


def centile_of(models, d, var):
    """Whether each value lies below each centile of its own age x sex curve (returns a DataFrame: one True/False column per q)."""
    rows = []
    for s, m in models.items():
        g = d[(d.sex == s)][['age', var]].dropna()
        if len(g) == 0:
            continue
        pr = predict(m, g.age.values)
        pr.index = g.index
        rows.append(pd.DataFrame({q: g[var].values < pr[q].values for q in pr.columns}, index=g.index))
    return pd.concat(rows)
