"""Extract intraoperative vital signs from MOVER flowsheets_cleaned, one parquet per input file (resumable)."""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import duckdb, glob, os, sys
SRC = _os.path.join(_MOVER, 'flowsheets_cleaned')
DST = _os.path.join(_PROJ, 'out', 'mover_intraop')
os.makedirs(DST, exist_ok=True)
ITEMS = ['Heart Rate', 'Pulse', 'SpO2', 'NIBP', 'NIBP - MAP', 'BP', 'MAP (mmHg)',
         'ETCO2 (mmHg)', 'ETCO2', 'Temp', 'Arterial Line BP (ART)', 'Arterial Line MAP (ART)',
         'BP-ART A-line', 'MAP-ART A-line', 'Resp', 'BIS']
items = ",".join("'" + i.replace("'", "''") + "'" for i in ITEMS)
con = duckdb.connect()
con.execute("set memory_limit='3GB'; set threads=4; set preserve_insertion_order=false")
for f in sorted(glob.glob(f'{SRC}/flowsheet_part*.csv')):
    out = f"{DST}/{os.path.basename(f).replace('.csv', '.parquet')}"
    if os.path.exists(out):
        continue
    con.execute(f"""copy (select LOG_ID, MRN, FLO_NAME, trim(FLO_DISPLAY_NAME) item,
                  try_cast(RECORDED_TIME as timestamp) t, MEAS_VALUE v, UNITS
                  from read_csv_auto('{f}', all_varchar=true)
                  where RECORD_TYPE='INTRA-OP' and trim(FLO_DISPLAY_NAME) in ({items}))
                  to '{out}.tmp' (format parquet)""")
    os.rename(out + '.tmp', out)
    print(f, 'done', flush=True)
