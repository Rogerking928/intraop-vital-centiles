"""VitalDB: per-minute medians during anaesthesia (anestart to aneend), written to a single parquet."""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, sys, numpy as np, pandas as pd, vitaldb
from multiprocessing import Pool
VD = _VITALDB
OUT = _os.path.join(_PROJ, 'out', 'vitaldb_minute.parquet')
TR = {'Solar8000/HR': 'hr', 'Solar8000/PLETH_SPO2': 'spo2', 'Solar8000/NIBP_SBP': 'nibp_sbp',
      'Solar8000/NIBP_DBP': 'nibp_dbp', 'Solar8000/NIBP_MBP': 'nibp_map', 'Solar8000/ART_SBP': 'art_sbp',
      'Solar8000/ART_DBP': 'art_dbp', 'Solar8000/ART_MBP': 'art_map', 'Solar8000/ETCO2': 'etco2',
      'Solar8000/BT': 'temp', 'BIS/BIS': 'bis'}
cases = pd.read_csv(f'{VD}/cases.csv').set_index('caseid')

def one(cid):
    try:
        vf = vitaldb.VitalFile(f'{VD}/vital/{cid}.vital', list(TR))
        names = [n for n in TR if n in vf.get_track_names()]
        if not names:
            return None
        a = vf.to_numpy(names, 2)            # one sample every 2 s, t=0 at casestart
        df = pd.DataFrame(a, columns=[TR[n] for n in names])
        df['minute'] = (np.arange(len(df)) * 2) // 60
        s, e = cases.at[cid, 'anestart'], cases.at[cid, 'aneend']
        df = df[(df.minute * 60 >= s) & (df.minute * 60 < e)]
        m = df.groupby('minute').median()
        m['min_from_anestart'] = m.index - int(s // 60)
        m['caseid'] = cid
        return m.reset_index(drop=True)
    except Exception as ex:
        print(cid, 'ERR', ex, file=sys.stderr, flush=True)
        return None

if __name__ == '__main__':
    with Pool(4) as p:
        res = [r for r in p.imap_unordered(one, cases.index, chunksize=8) if r is not None]
    pd.concat(res).to_parquet(OUT)
    print('done', len(res))
