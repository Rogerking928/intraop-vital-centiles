"""逐筆 NIBP 讀值 → 每人的讀值層級指標，兩個時窗：
  maint：麻醉開始 +15 分到結束 −15 分（主分析時窗）
  full ：麻醉開始到結束（含誘導期）
指標：中位數、時間加權平均（TWA，每筆讀值算到下一筆、上限 10 分鐘）、
      持續 5 分鐘的最低 MAP（相鄰讀值涵蓋 ≥5 分鐘時，該段內的最高值；取全程最小）＝ lowest sustained MAP、
      單筆最低 MAP、低於 65 的分鐘數。
輸出 out/readings_mover.parquet（每人一列，兩個時窗各一組欄位）。
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
    t = g.t.values.astype('datetime64[s]').astype(np.int64) / 60.0
    x = g.mp.values
    dur = np.minimum(np.append(np.diff(t), 5.0), 10.0)
    out = {'n': len(x), 'median': np.median(x), 'twa': np.sum(x * dur) / np.sum(dur), 'min1': x.min(),
           'min_lt65': float(np.sum(dur[x < 65]))}
    # 持續 5 分鐘：從第 i 筆起，往後取到涵蓋 ≥5 分鐘為止，該段最大值；全程最小
    best = np.nan
    j = 0
    for i in range(len(x)):
        j = max(j, i)
        while j < len(x) - 1 and t[j] - t[i] < 5:
            j += 1
        if t[j] - t[i] >= 5 or (j == len(x) - 1 and t[j] - t[i] + dur[j] >= 5):
            v = x[i:j + 1].max()
            best = v if np.isnan(best) else min(best, v)
    out['sust5'] = best
    return pd.Series(out)


res = []
for win, sub in (('maint', r[r.maint]), ('full', r)):
    m = sub.groupby('LOG_ID', sort=False).apply(metrics)
    m.columns = [f'{win}_{c}' for c in m.columns]
    res.append(m)
out = pd.concat(res, axis=1).reset_index()
out.to_parquet(f'{P}/out/readings_mover.parquet')
print(out.describe().round(1).T)
