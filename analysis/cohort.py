"""Build a one-row-per-patient analysis table for each database, used by norms.py and validate.py.

MOVER: adults (18-90), general anaesthesia, anaesthesia >=60 min, first case per patient; excludes cardiac surgery and
cardiac catheterisation, craniotomy, and pregnancy-related surgery.
Maintenance phase = anaesthesia start +15 min to anaesthesia end -15 min. Uses the monitor auto-upload rows
(FLO_NAME='Devices Testing Template').
NIBP artefact rules follow de Graaff 2016: pulse pressure <=5, DBP>MAP, MAP>SBP or SBP >=250 is an artefact (whole set dropped).
VitalDB: same criteria; per-minute medians come from extract_vitaldb.py.

Outputs out/cohort_mover.parquet, out/cohort_vitaldb.parquet, out/cohort_flow.json
"""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, json, duckdb, numpy as np, pandas as pd

P = _PROJ
MX = _os.path.join(_MOVER, 'EPIC_EMR/EMR')
VD = _VITALDB
con = duckdb.connect()
con.execute("set memory_limit='4GB'; set threads=6")
CARDIAC = ('CABG|CORONARY ARTERY BYPASS|VALVE|VALVULOPLASTY|CATHETERIZATION, HEART|HEART CATH|TAVR|LVAD|'
           'STERNOTOMY|OPEN HEART|ASCENDING AORTA|AORTIC ROOT|AORTIC ARCH|INTRA-AORTIC BALLOON|PROCUREMENT, HEART|'
           'TRANSPLANT, HEART|HEART TRANSPLANT|CARDIOPULMONARY BYPASS|PCI|CARDIAC ELECTROPHYSIOLOGY|ABLATION, CARDIAC')
# Also not used as reference: surgery where vital signs are deliberately held at non-physiological targets
# (craniotomy: hyperventilation, blood-pressure targets) or physiology differs (pregnancy)
INTRACRANIAL = 'CRANIOTOMY|CRANIECTOMY|BURR HOLE|TRANSSPHENOIDAL|INTRACRANIAL'
OBSTETRIC = 'CESAREAN|DILATION AND EVACUATION, UTERUS|ECTOPIC PREGNANCY|POSTPARTUM'

# ---------------- MOVER ----------------
pi = con.execute(f"""
select LOG_ID, MRN, try_cast(BIRTH_DATE as int) age, SEX sex, PRIMARY_ANES_TYPE_NM anes,
       try_cast(ASA_RATING_C as double)::int asa, PATIENT_CLASS_NM cls, PRIMARY_PROCEDURE_NM proc,
       try_cast(WEIGHT as double) * 0.0283495 wt_kg, HEIGHT ht,
       try_strptime(AN_START_DATETIME, '%m/%d/%y %H:%M') an0, try_strptime(AN_STOP_DATETIME, '%m/%d/%y %H:%M') an1
from read_csv_auto('{MX}/patient_information.csv', all_varchar=true)""").df()
flow = {'records': len(pi), 'cases': int(pi.LOG_ID.nunique())}
# A few LOG_IDs (8) have conflicting duplicate rows; always take the row with the latest anaesthesia start, for reproducibility
pi = pi.sort_values(['LOG_ID', 'an0', 'MRN', 'an1', 'proc', 'asa'], ascending=[True, False, True, True, True, True],
                    kind='mergesort', na_position='last').drop_duplicates('LOG_ID')
pi = pi[(pi.age >= 18) & (pi.anes == 'General') & pi.sex.isin(['Female', 'Male'])]
flow['adult_general'] = len(pi)
pu = pi.proc.fillna('').str.upper()
pi['cardiac'] = pu.str.contains(CARDIAC)
pi['intracranial'] = pu.str.contains(INTRACRANIAL) & ~pi.cardiac
pi['obstetric'] = pu.str.contains(OBSTETRIC) & ~pi.cardiac & ~pi.intracranial
flow['excl_cardiac'] = int(pi.cardiac.sum())
flow['excl_intracranial'] = int(pi.intracranial.sum())
flow['excl_obstetric'] = int(pi.obstetric.sum())
pi = pi[~(pi.cardiac | pi.intracranial | pi.obstetric)]
flow['non_cardiac'] = len(pi)
pi['dur'] = (pi.an1 - pi.an0).dt.total_seconds() / 60
pi = pi[pi.dur >= 60]
flow['anes_ge_60min'] = len(pi)
# Multiple records for the same patient and start time: take the smallest LOG_ID (fixed rule)
pi = pi.sort_values(['MRN', 'an0', 'LOG_ID'], kind='mergesort').drop_duplicates('MRN')
flow['first_case_per_patient'] = len(pi)


def height_m(s):
    try:
        ft, inch = str(s).replace('"', '').split("'")
        return (int(ft) * 12 + float(inch or 0)) * 0.0254
    except Exception:
        return np.nan


pi['ht_m'] = pi.ht.map(height_m)
pi['bmi'] = pi.wt_kg / pi.ht_m ** 2
pi.loc[(pi.bmi < 12) | (pi.bmi > 80), 'bmi'] = np.nan
con.register('pi', pi[['LOG_ID', 'an0', 'an1']])

# Maintenance-phase readings (NIBP SBP/DBP/MAP paired by identical timestamp)
con.execute(f"""
create temp table v as
select f.LOG_ID, f.item, f.t, f.v from '{P}/out/mover_intraop/*.parquet' f join pi using (LOG_ID)
where f.FLO_NAME = 'Devices Testing Template'
  and f.t >= pi.an0 + interval 15 minute and f.t < pi.an1 - interval 15 minute""")
con.execute("""
create temp table nibp as
select a.LOG_ID, a.t, try_cast(split_part(a.v,'/',1) as double) sbp, try_cast(split_part(a.v,'/',2) as double) dbp,
       try_cast(b.v as double) mp
from (select * from v where item='NIBP') a join (select * from v where item='NIBP - MAP') b using (LOG_ID, t)""")
con.execute("""
create temp table nibp_ok as select * from nibp
where sbp < 250 and sbp - dbp > 5 and dbp <= mp and mp <= sbp and mp between 30 and 200 and dbp >= 10""")
flow['nibp_triplets'] = con.execute("select count(*) from nibp").fetchone()[0]
flow['nibp_triplets_valid'] = con.execute("select count(*) from nibp_ok").fetchone()[0]
nb = con.execute("""
select LOG_ID, median(mp) nibp_map, median(sbp) nibp_sbp, median(dbp) nibp_dbp, count(*) n_nibp,
       avg((mp < 65)::int) nibp_frac_lt65, min(mp) nibp_min_map
from nibp_ok group by 1""").df()
ot = con.execute("""
select LOG_ID,
  median(case when item='Heart Rate' then x end) hr,  count(case when item='Heart Rate' then 1 end) n_hr,
  median(case when item='SpO2' then x end) spo2,
  median(case when item='ETCO2 (mmHg)' then x end) etco2,
  median(case when item='MAP-ART A-line' then x end) art_map, count(case when item='MAP-ART A-line' then 1 end) n_art,
  median(case when item='Temp' then (x-32)*5/9 end) temp_c
from (select LOG_ID, item, try_cast(v as double) x from v
      where item in ('Heart Rate','SpO2','ETCO2 (mmHg)','MAP-ART A-line','Temp'))
where (item='Heart Rate' and x between 20 and 250) or (item='SpO2' and x between 50 and 100)
   or (item='ETCO2 (mmHg)' and x between 10 and 90) or (item='MAP-ART A-line' and x between 30 and 200)
   or (item='Temp' and x between 86 and 108)
group by 1""").df()

# Intraoperative vasoactive drugs: infusions (New Bag/Rate Change/Restarted) and boluses (Given)
va = con.execute(f"""
select LOG_ID,
  max((regexp_matches(upper(MEDICATION_NM),'INFUSION|/250 ML|/100 ML') and MAR_ACTION_NM in ('New Bag','Rate Change','Restarted'))::int) vaso_inf,
  max((not regexp_matches(upper(MEDICATION_NM),'INFUSION|LIDOCAINE|BUPIVACAINE|ROPIVACAINE|NASAL|TOPICAL|ARTICAINE|PRILOCAINE|MEPIVACAINE|CHLOROPROCAINE')
       and MAR_ACTION_NM='Given')::int) vaso_bolus
from '{P}/out/mover_vasoactive_intraop.parquet'
where not regexp_matches(upper(MEDICATION_NM),'LIDOCAINE|BUPIVACAINE|ROPIVACAINE|NASAL|TOPICAL|ARTICAINE|PRILOCAINE|MEPIVACAINE|CHLOROPROCAINE')
group by 1""").df()

mv = pi.merge(nb, on='LOG_ID', how='left').merge(ot, on='LOG_ID', how='left').merge(va, on='LOG_ID', how='left')
mv[['vaso_inf', 'vaso_bolus']] = mv[['vaso_inf', 'vaso_bolus']].fillna(0).astype(int)
mv = mv[mv.n_hr.notna()]
flow['has_intraop_vitals'] = len(mv)
mv = mv[(mv.n_hr >= 20) & (mv.n_nibp >= 3)]
flow['analysable'] = len(mv)
mv['ref'] = mv.asa.isin([1, 2])
flow['reference_asa12'] = int(mv.ref.sum())
flow['asa_missing'] = int(mv.asa.isna().sum())
mv = mv.drop(columns=['ht', 'anes', 'MRN'])
OD = os.environ.get('COHORT_OUT', f'{P}/out')
mv.to_parquet(f'{OD}/cohort_mover.parquet')

# ---------------- VitalDB ----------------
vc = pd.read_csv(f'{VD}/cases.csv')
trk = pd.read_csv(f'{VD}/trks.csv')
inf_ids = set(trk[trk.tname.str.match(r'Orchestra/(PHEN|NEPI|EPI|VASO|DOPA|DOBU)_RATE')].caseid)
vm = pd.read_parquet(f'{P}/out/vitaldb_minute.parquet').rename(columns={'temp': 'temp_c'})
end = vc.set_index('caseid').eval('(aneend-anestart)/60')
vm = vm[(vm.min_from_anestart >= 15) & (vm.min_from_anestart < vm.caseid.map(end) - 15)]
rng = {'hr': (20, 250), 'spo2': (50, 100), 'etco2': (10, 90), 'art_map': (30, 200), 'temp_c': (30, 42)}
for k, (lo, hi) in rng.items():
    vm.loc[(vm[k] < lo) | (vm[k] > hi), k] = np.nan
ok = ((vm.nibp_sbp < 250) & (vm.nibp_sbp - vm.nibp_dbp > 5) & (vm.nibp_dbp <= vm.nibp_map) &
      (vm.nibp_map <= vm.nibp_sbp) & vm.nibp_map.between(30, 200) & (vm.nibp_dbp >= 10))
for k in ['nibp_sbp', 'nibp_dbp', 'nibp_map']:
    vm.loc[~ok, k] = np.nan
g = vm.groupby('caseid')
vmed = g[list(rng) + ['nibp_sbp', 'nibp_dbp', 'nibp_map']].median()
vmed['n_hr'] = g.hr.count()
vmed['n_nibp'] = g.nibp_map.count()
vmed['n_art'] = g.art_map.count()
vmed['nibp_frac_lt65'] = g.nibp_map.apply(lambda s: (s.dropna() < 65).mean() if s.notna().any() else np.nan)
vv = vc.set_index('caseid').join(vmed, how='inner')
vflow = {'cases': len(vc), 'with_minute_data': len(vv)}
vv = vv[(vv.age >= 18) & (vv.ane_type == 'General') & vv.sex.isin(['F', 'M'])]
vflow['adult_general'] = len(vv)
vo = vv.opname.fillna('').str.upper()
vflow['excl_obstetric'] = int(vo.str.contains('CESAREAN').sum())
vflow['excl_intracranial'] = int(vo.str.contains('CRANIOTOMY|CRANIECTOMY|BURR HOLE|TRANSSPHENOIDAL|INTRACRANIAL').sum())
vv = vv[~vo.str.contains('CESAREAN|CRANIOTOMY|CRANIECTOMY|BURR HOLE|TRANSSPHENOIDAL|INTRACRANIAL')]
vflow['non_cardiac'] = len(vv)
vv['dur'] = (vv.aneend - vv.anestart) / 60
vv = vv[vv.dur >= 60]
vflow['anes_ge_60min'] = len(vv)
vv = vv.sort_values(['subjectid', 'casestart', 'caseid'], kind='mergesort').drop_duplicates('subjectid')
vflow['first_case_per_subject'] = len(vv)
vv = vv[vv.n_hr >= 20]
vflow['analysable_hr'] = len(vv)
vv['sex'] = vv.sex.map({'F': 'Female', 'M': 'Male'})
vv['vaso_inf'] = vv.index.isin(inf_ids).astype(int)
vv['vaso_bolus'] = ((vv.intraop_phe.fillna(0) > 0) | (vv.intraop_eph.fillna(0) > 0)).astype(int)
vv['ref'] = vv.asa.isin([1, 2]) & (vv.emop != 1)
vflow['reference_asa12_elective'] = int(vv.ref.sum())
vflow['asa_missing'] = int(vv.asa.isna().sum())
vv = vv.reset_index()[['caseid', 'age', 'sex', 'asa', 'emop', 'department', 'optype', 'approach', 'bmi', 'dur',
                       'vaso_inf', 'vaso_bolus', 'ref', 'hr', 'spo2', 'etco2', 'art_map', 'temp_c', 'nibp_sbp',
                       'nibp_dbp', 'nibp_map', 'n_hr', 'n_nibp', 'n_art', 'nibp_frac_lt65']]
vv.to_parquet(f'{OD}/cohort_vitaldb.parquet')

json.dump({'mover': flow, 'vitaldb': vflow}, open(f'{OD}/cohort_flow.json', 'w'), indent=1)
print(json.dumps({'mover': flow, 'vitaldb': vflow}, indent=1))
print(mv.groupby('ref')[['vaso_inf', 'vaso_bolus']].mean())
print(vv.groupby('ref')[['vaso_inf', 'vaso_bolus']].mean())
print(mv.cls.value_counts())
