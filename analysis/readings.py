"""Individual NIBP readings -> per-patient reading-level metrics, in two windows:
  maint: anaesthesia start +15 min to end -15 min (primary analysis window)
  full : anaesthesia start to end (including induction)
Metrics: median; time-weighted average (TWA; each reading lasts until the next, capped at 10 min);
      lowest MAP sustained for 5 min (for consecutive readings spanning >=5 min, the highest value within that span;
      minimum over the case) = lowest sustained MAP; lowest single MAP; minutes below 65.
Reading durations are always computed on the full-anaesthesia series (until the next reading, capped at 10 min),
so the two windows are consistent.
Outputs out/readings_mover.parquet (one row per patient, one set of columns per window).
"""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, duckdb, numpy as np, pandas as pd

P = _PROJ
con = duckdb.connect()
con.execute("set memory_limit='4GB'; set threads=6")
mv = pd.read_parquet(f'{P}/out/cohort_mover.parquet')[['LOG_ID', 'an0', 'an1']]
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
select LOG_ID, t, mp, (t >= an0 + interval 15 minute and t < an1 - interval 15 minute) maint
from a where sbp < 250 and sbp - dbp > 5 and dbp <= mp and mp <= sbp and mp between 30 and 200 and dbp >= 10
order by LOG_ID, t""").df()


def metrics(g):
    t = g.tmin.values
    x = g.mp.values
    dur = g.dur.values                     # interval to the next reading of the WHOLE anaesthetic, capped at 10 min
    out = {'n': len(x), 'median': np.median(x), 'twa': np.sum(x * dur) / np.sum(dur), 'min1': x.min(),
           'min_lt65': float(np.sum(dur[x < 65]))}
    # lowest sustained 5 min: the highest value among consecutive readings whose timestamps span at least 5 min
    # (so at least two readings); take the minimum over all such runs. A run inside the maintenance window is also a
    # run of the whole anaesthetic, so the whole-anaesthetic value can never exceed the maintenance value.
    best, j = np.nan, 0
    for i in range(len(x)):
        j = max(j, i)
        while j < len(x) - 1 and t[j] - t[i] < 5:
            j += 1
        if t[j] - t[i] >= 5:
            v = x[i:j + 1].max()
            best = v if np.isnan(best) else min(best, v)
    out['sust5'] = best
    return pd.Series(out)


r['tmin'] = r.t.values.astype('datetime64[s]').astype(np.int64) / 60.0
nxt = r.groupby('LOG_ID', sort=False).tmin.shift(-1)
r['dur'] = np.minimum((nxt - r.tmin).fillna(5.0), 10.0)
res = []
for win, sub in (('maint', r[r.maint]), ('full', r)):
    m = sub.groupby('LOG_ID', sort=False).apply(metrics)
    m.columns = [f'{win}_{c}' for c in m.columns]
    res.append(m)
out = pd.concat(res, axis=1).reset_index()
chk = out.full_sust5 > out.maint_sust5 + 1e-9
assert not chk.any(), f"{int(chk.sum())} patients with a whole-anaesthesia sustained MAP above the maintenance one"
out.to_parquet(f'{P}/out/readings_mover.parquet')
print(out.describe().round(1).T)
