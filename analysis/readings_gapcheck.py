"""Sensitivity check for the lowest sustained 5-minute MAP (added after external review, 2026-10-02): the primary
definition places no limit on the interval between consecutive readings of a run. Here a run is also broken at any
interval over 10 minutes (the cap used for the time-weighted average). Reports, for the reference cohort (ASA I-II),
the share of patients whose maintenance value changes and by how much. Output out/readings_gapcheck.json."""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, json, duckdb, numpy as np, pandas as pd

P = _PROJ
con = duckdb.connect()
con.execute("set memory_limit='4GB'; set threads=4")
mv = pd.read_parquet(f'{P}/out/cohort_mover.parquet')
mv = mv[mv.ref][['LOG_ID', 'an0', 'an1']]
con.register('pi', mv)
r = con.execute(f"""
with a as (
  select a.LOG_ID, a.t, try_cast(split_part(a.v,'/',1) as double) sbp, try_cast(split_part(a.v,'/',2) as double) dbp,
         try_cast(b.v as double) mp, pi.an0, pi.an1
  from '{P}/out/mover_intraop/*.parquet' a
  join '{P}/out/mover_intraop/*.parquet' b on a.LOG_ID=b.LOG_ID and a.t=b.t
  join pi on a.LOG_ID = pi.LOG_ID
  where a.FLO_NAME='Devices Testing Template' and b.FLO_NAME='Devices Testing Template'
    and a.item='NIBP' and b.item='NIBP - MAP' and a.t >= pi.an0 and a.t <= pi.an1)
select LOG_ID, t, mp from a
where sbp < 250 and sbp - dbp > 5 and dbp <= mp and mp <= sbp and mp between 30 and 200 and dbp >= 10
  and t >= an0 + interval 15 minute and t < an1 - interval 15 minute
order by LOG_ID, t""").df()
r['tmin'] = r.t.values.astype('datetime64[s]').astype(np.int64) / 60.0


def sust(t, x, maxgap=None):
    best, j = np.nan, 0
    for i in range(len(x)):
        j = max(j, i)
        while j < len(x) - 1 and t[j] - t[i] < 5 and (maxgap is None or t[j + 1] - t[j] <= maxgap):
            j += 1
        if t[j] - t[i] >= 5 and (maxgap is None or np.all(np.diff(t[i:j + 1]) <= maxgap)):
            v = x[i:j + 1].max()
            best = v if np.isnan(best) else min(best, v)
    return best


res = r.groupby('LOG_ID', sort=False).apply(lambda g: pd.Series({'primary': sust(g.tmin.values, g.mp.values),
                                                                  'gap10': sust(g.tmin.values, g.mp.values, 10.0)}))
old = pd.read_parquet(f'{P}/out/readings_mover.parquet').set_index('LOG_ID').maint_sust5
assert np.allclose(res.primary, old.reindex(res.index), equal_nan=True), "primary definition not reproduced"
d = (res.gap10 - res.primary)
out = {'patients': int(len(res)), 'changed': int((d.abs() > 1e-9).sum()), 'changed_pct': float(100 * (d.abs() > 1e-9).mean()),
       'undefined_with_gap10': int(res.gap10.isna().sum()),
       'median_change_if_changed': float(d[d.abs() > 1e-9].median()) if (d.abs() > 1e-9).any() else 0.0,
       'below65_primary_pct': float(100 * (res.primary < 65).mean()), 'below65_gap10_pct': float(100 * (res.gap10 < 65).mean())}
json.dump(out, open(f'{P}/out/readings_gapcheck.json', 'w'), indent=1)
print(out)
