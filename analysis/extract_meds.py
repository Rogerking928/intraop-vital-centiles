"""Intraoperative vasoactive and propofol administrations from the MOVER medication table (patient_medications.csv).
Outputs out/mover_vasoactive_intraop.parquet (read by cohort.py) and out/mover_propofol_intraop.parquet (read by
revision.py). Run after extract_mover.py and before cohort.py."""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, duckdb

P = _PROJ
MED = _os.path.join(_MOVER, 'EPIC_EMR/EMR/patient_medications.csv')
con = duckdb.connect()
con.execute("set memory_limit='4GB'; set threads=4")
con.execute(f"""copy (select LOG_ID, MEDICATION_NM, MAR_ACTION_NM, MED_ACTION_TIME, ADMIN_SIG, DOSE_UNIT_NM, MED_ROUTE_NM
    from read_csv_auto('{MED}', all_varchar=true) where RECORD_TYPE='INTRA-OP'
    and regexp_matches(upper(MEDICATION_NM), 'PHENYLEPHRINE|NOREPINEPHRINE|EPINEPHRINE|VASOPRESSIN|DOPAMINE|EPHEDRINE|DOBUTAMINE'))
    to '{P}/out/mover_vasoactive_intraop.parquet' (format parquet)""")
con.execute(f"""copy (select LOG_ID, MEDICATION_NM, MAR_ACTION_NM, MED_ACTION_TIME
    from read_csv_auto('{MED}', all_varchar=true) where RECORD_TYPE='INTRA-OP' and regexp_matches(upper(MEDICATION_NM), 'PROPOFOL'))
    to '{P}/out/mover_propofol_intraop.parquet' (format parquet)""")
for f in ('mover_vasoactive_intraop', 'mover_propofol_intraop'):
    print(f, con.execute(f"select count(*) from '{P}/out/{f}.parquet'").fetchone()[0])
