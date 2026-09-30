"""MOVER flowsheets_cleaned → 術前血壓（PRE-OP 的 BP／MAP／NIBP）與術中吸入麻醉藥（ET 濃度項目），逐檔寫 parquet。"""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import duckdb, glob, os
SRC = _os.path.join(_MOVER, 'flowsheets_cleaned')
DST = _os.path.join(_PROJ, 'out', 'mover_preop')
os.makedirs(DST, exist_ok=True)
con = duckdb.connect()
con.execute("set memory_limit='3GB'; set threads=4; set preserve_insertion_order=false")
for f in sorted(glob.glob(f'{SRC}/flowsheet_part*.csv')):
    out = f"{DST}/{os.path.basename(f).replace('.csv', '.parquet')}"
    if os.path.exists(out):
        continue
    con.execute(f"""copy (select LOG_ID, RECORD_TYPE rt, FLO_NAME, trim(FLO_DISPLAY_NAME) item,
                  try_cast(RECORDED_TIME as timestamp) t, MEAS_VALUE v
                  from read_csv_auto('{f}', all_varchar=true)
                  where (RECORD_TYPE='PRE-OP' and trim(FLO_DISPLAY_NAME) in ('BP','MAP (mmHg)','NIBP','NIBP - MAP'))
                     or (RECORD_TYPE='INTRA-OP' and regexp_matches(lower(FLO_DISPLAY_NAME), 'sev|iso |des |iso et|des et|etn2o|agent')))
                  to '{out}.tmp' (format parquet)""")
    os.rename(out + '.tmp', out)
    print(f, 'done', flush=True)
