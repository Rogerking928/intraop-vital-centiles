"""Table 1：MOVER 參考族群、MOVER 全部可分析者、VitalDB 驗證族群的特徵。輸出 out/t1_characteristics.csv"""
import os as _os
_PROJ = _os.environ.get('PROJECT_DIR', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MOVER = _os.environ.get('MOVER_DIR', _os.path.join(_PROJ, 'data', 'MOVER'))
_VITALDB = _os.environ.get('VITALDB_DIR', _os.path.join(_PROJ, 'data', 'VitalDB'))
import os, numpy as np, pandas as pd

P = _PROJ
mv = pd.read_parquet(f'{P}/out/cohort_mover.parquet')
vd = pd.read_parquet(f'{P}/out/cohort_vitaldb.parquet')
for k in ['nibp_map', 'nibp_sbp', 'nibp_dbp']:
    vd.loc[vd.n_nibp < 3, k] = np.nan
mv['has_art'] = mv.n_art.fillna(0) > 20        # MOVER：動脈導管讀值 >20 筆（約每分鐘一筆）
vd['has_art'] = vd.n_art.fillna(0) > 20        # VitalDB：ART 訊號 >20 分鐘
cols = {'MOVER reference (ASA I-II)': mv[mv.ref], 'MOVER, all ASA classes': mv[mv.asa.notna()],
        'VitalDB validation (ASA I-II, elective)': vd[vd.ref]}


def med(x, d=0):
    x = x.dropna()
    return f"{x.median():.{d}f} ({x.quantile(.25):.{d}f}-{x.quantile(.75):.{d}f})"


def pct(mask, n):
    return f"{int(mask.sum()):,} ({100 * mask.sum() / n:.1f})"


rows = []
def add(label, f):
    rows.append({'row': label, **{c: f(d) for c, d in cols.items()}})


add('Patients, n', lambda d: f"{len(d):,}")
add('Age, years', lambda d: med(d.age))
add('Female sex', lambda d: pct(d.sex == 'Female', len(d)))
add('Body mass index, kg/m2', lambda d: med(d.bmi, 1))
for a in (1, 2, 3):
    add(f'ASA physical status {"I" * a if a < 4 else ""}'.replace('III', 'III'), lambda d, a=a: pct(d.asa == a, len(d)))
add('ASA physical status IV-VI', lambda d: pct(d.asa >= 4, len(d)))
add('Anaesthesia duration, min', lambda d: med(d.dur))
add('Arterial line', lambda d: pct(d.has_art, len(d)))
add('Vasoactive infusion', lambda d: pct(d.vaso_inf == 1, len(d)))
add('Vasopressor bolus', lambda d: pct(d.vaso_bolus == 1, len(d)))
add('MOVER: outpatient surgery', lambda d: pct(d.cls == 'Hospital Outpatient Surgery', len(d)) if 'cls' in d else '')
for dep in ['General surgery', 'Thoracic surgery', 'Gynecology', 'Urology']:
    add(f'VitalDB department: {dep}', lambda d, dep=dep: pct(d.department == dep, len(d)) if 'department' in d else '')
add('Case median MAP, mmHg', lambda d: med(d.nibp_map))
add('Case median heart rate, beats/min', lambda d: med(d.hr))
add('Case median end-tidal CO2, mmHg', lambda d: med(d.etco2))
add('Case median temperature, °C', lambda d: med(d.temp_c, 1))
add('Temperature recorded', lambda d: pct(d.temp_c.notna(), len(d)))
add('Non-invasive pressure recorded (≥3 readings)', lambda d: pct(d.nibp_map.notna(), len(d)))
t = pd.DataFrame(rows)
t.to_csv(f'{P}/out/t1_characteristics.csv', index=False)
print(t.to_string())
