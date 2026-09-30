"""Supplementary_Calculator.html：單一離線 HTML，內嵌 norms_table.csv（參考族群 ASA I–II 的每歲百分位）。
輸入性別、年齡、變數與數值 → 在該年齡×性別的七個百分位之間線性內插出百分位，並列出 P3–P97。"""
import os, json, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t = pd.read_csv(f"{ROOT}/out/norms_table.csv")
data = {}
for (v, s), g in t.groupby(['var', 'sex']):
    data.setdefault(v, {})[s] = {int(r.age): [round(r[c], 2) for c in ['P3', 'P10', 'P25', 'P50', 'P75', 'P90', 'P97']]
                                 for _, r in g.iterrows()}
html = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Intraoperative Vital Sign Centiles</title>
<style>
:root{--bg:#fbfaf7;--fg:#1f2933;--mut:#5b6b7a;--line:#d9d6cf;--acc:#2a78d6;--card:#ffffff}
@media (prefers-color-scheme:dark){:root{--bg:#15191d;--fg:#e8e6e1;--mut:#9aa5b1;--line:#333a41;--acc:#6da7ec;--card:#1d2227}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:680px;margin:0 auto;padding:24px 16px}
h1{font-size:1.3rem;margin:0 0 4px}p.note{color:var(--mut);font-size:.85rem;margin:0 0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
label{display:block;font-size:.8rem;color:var(--mut);margin-bottom:4px}
select,input{width:100%;box-sizing:border-box;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);font-size:1rem}
.out{margin-top:20px;padding:16px;border:1px solid var(--line);border-radius:8px;background:var(--card)}
.big{font-size:2rem;font-weight:600;color:var(--acc)}
table{width:100%;border-collapse:collapse;margin-top:12px;font-size:.9rem}td,th{padding:4px 6px;text-align:center;border-bottom:1px solid var(--line)}
</style></head><body><main>
<h1>Centiles of vital signs during general anaesthesia</h1>
<p class="note">Reference: adults with ASA physical status I-II, patient median over the maintenance phase of general anaesthesia (15 min after induction to 15 min before the end), MOVER, University of California, Irvine. Non-invasive pressures. Heart-rate and end-tidal CO2 centiles were miscalibrated in an external Korean cohort and should be recalibrated locally. These centiles describe usual practice; they are not thresholds for harm and must not replace clinical judgement.</p>
<div class="grid">
<div><label for="sex">Sex</label><select id="sex"><option>Female</option><option>Male</option></select></div>
<div><label for="age">Age (18-90 years)</label><input id="age" type="number" min="18" max="90" value="30"></div>
<div><label for="var">Variable</label><select id="var">
<option value="nibp_map">Mean arterial pressure (mm Hg)</option><option value="nibp_sbp">Systolic pressure (mm Hg)</option>
<option value="nibp_dbp">Diastolic pressure (mm Hg)</option><option value="hr">Heart rate (beats/min)</option>
<option value="etco2">End-tidal CO2 (mm Hg)</option><option value="temp_c">Temperature (°C)</option></select></div>
<div><label for="val">Patient value</label><input id="val" type="number" step="0.1" value="65"></div>
</div>
<div class="out"><div id="res" class="big"></div><div id="txt" class="note"></div>
<table><thead><tr><th>P3</th><th>P10</th><th>P25</th><th>P50</th><th>P75</th><th>P90</th><th>P97</th></tr></thead><tbody><tr id="row"></tr></tbody></table></div>
</main><script>
const D = __DATA__;
const Q = [3,10,25,50,75,90,97];
function go(){
  const s=document.getElementById('sex').value, v=document.getElementById('var').value;
  let a=Math.round(+document.getElementById('age').value); a=Math.min(90,Math.max(18,a||18));
  const x=+document.getElementById('val').value, c=D[v][s][a];
  document.getElementById('row').innerHTML=c.map(z=>'<td>'+z.toFixed(v==='temp_c'?1:0)+'</td>').join('');
  let r;
  if(!isFinite(x)){r=''} else if(x<c[0]){r='below the 3rd centile'} else if(x>c[6]){r='above the 97th centile'}
  else{for(let i=0;i<6;i++){if(x<=c[i+1]){r=(Q[i]+(x-c[i])/(c[i+1]-c[i]||1)*(Q[i+1]-Q[i])).toFixed(0)+'th centile (approx.)';break}}}
  document.getElementById('res').textContent=r;
  document.getElementById('txt').textContent=s+', '+a+' years';
}
['sex','age','var','val'].forEach(id=>document.getElementById(id).addEventListener('input',go));go();
</script></body></html>'''
open(f"{ROOT}/manuscript/Supplementary_Calculator.html", "w").write(html.replace("__DATA__", json.dumps(data, separators=(',', ':'))))
print('written')
