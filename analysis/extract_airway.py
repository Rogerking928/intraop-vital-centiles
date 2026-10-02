"""Intraoperative airway or oxygen device from the MOVER flowsheets (added after external review, 2026-10-02, to
examine whether cases with spontaneous ventilation or sedation recorded as general anaesthesia lower the end-tidal CO2
centiles). Extracts INTRA-OP 'O2 Device' and 'Airway Device' entries. Output out/mover_airway_raw.parquet."""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import duckdb, glob, os
FS = _os.path.join(_MOVER, 'flowsheets_cleaned')
P = _PROJ
OUT = f'{P}/out'
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
