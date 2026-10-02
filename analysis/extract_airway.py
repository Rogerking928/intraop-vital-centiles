"""術中氣道器材（外部審閱 2026-10-02：EtCO2 P3 偏低，疑似混入未插管的鎮靜個案）。
從 flowsheets_cleaned 抽 INTRA-OP 的 'O2 Device' 與 'Airway Device'，每個 LOG_ID 一列。輸出 out/mover_airway.parquet。"""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import duckdb, glob, os
FS = _os.path.join(_MOVER, 'flowsheets_cleaned')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
con = duckdb.connect(); con.execute("SET threads=4"); con.execute("SET memory_limit='6GB'")
parts = []
for f in sorted(glob.glob(f'{FS}/flowsheet_part*.csv')):
    d = con.execute(f"""select LOG_ID, FLO_DISPLAY_NAME item, MEAS_VALUE v, RECORDED_TIME t
        from read_csv('{f}', all_varchar=true, header=true)
        where RECORD_TYPE='INTRA-OP' and FLO_DISPLAY_NAME in ('O2 Device', 'Airway Device')""").df()
    parts.append(d); print(os.path.basename(f), len(d), flush=True)
import pandas as pd
a = pd.concat(parts, ignore_index=True)
a.to_parquet(f'{OUT}/mover_airway_raw.parquet')
print(a.groupby('item').v.value_counts().head(40))
