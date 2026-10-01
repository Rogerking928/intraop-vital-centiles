"""Figures at print width (174 mm), PNG at 700 dpi. House style of the asthma / Enterococcus papers:
DejaVu Sans, 9 pt text, 8.5 pt ticks, ink #1f2933, blue-grey axes, no grid lines.
Reads out/*.csv and out/*.parquet written by analysis/*.py; fails if any two text boxes overlap."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/out"
W = 174 / 25.4
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
                     'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
                     'axes.edgecolor': '#5b6b7a', 'axes.labelcolor': '#1f2933', 'text.color': '#1f2933',
                     'xtick.color': '#5b6b7a', 'ytick.color': '#5b6b7a', 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42})
INK, INK2, GRID = '#1f2933', '#5b6b7a', '#e4e3df'
SEXCOL = {'Female': '#eb6834', 'Male': '#2a78d6'}
LAB = {'nibp_map': 'MAP (mm Hg)', 'nibp_sbp': 'Systolic pressure (mm Hg)', 'nibp_dbp': 'Diastolic pressure (mm Hg)',
       'hr': 'Heart rate (beats min$^{-1}$)', 'etco2': 'End-tidal CO$_2$ (mm Hg)', 'temp_c': 'Temperature (°C)'}
TITLE = {'nibp_map': 'Mean arterial pressure', 'nibp_sbp': 'Systolic pressure', 'nibp_dbp': 'Diastolic pressure',
         'hr': 'Heart rate', 'etco2': 'End-tidal CO$_2$', 'temp_c': 'Temperature'}


def save(fig, name):
    bad = overlap_check(fig)
    assert not bad, f"{name}: overlapping text {bad}"
    fig.savefig(f"{OUT}/{name}.png", dpi=700, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def overlap_check(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
        xt = [t for t, v in zip(ax.get_xticklabels(), ax.get_xticks()) if x0 - 1e-9 <= v <= x1 + 1e-9]
        yt = [t for t, v in zip(ax.get_yticklabels(), ax.get_yticks()) if y0 - 1e-9 <= v <= y1 + 1e-9]
        for t in ax.texts + [ax.xaxis.label, ax.yaxis.label, ax.title] + xt + yt:
            if t.get_text() and t.get_visible():
                boxes.append((t.get_text(), t.get_window_extent(r)))
    for t in fig.texts:
        if t.get_text(): boxes.append((t.get_text(), t.get_window_extent(r)))
    k = 72 / fig.dpi
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            w = (min(a.x1, b.x1) - max(a.x0, b.x0)) * k
            h = (min(a.y1, b.y1) - max(a.y0, b.y0)) * k
            if w > 0.6 and h > 0.6:
                bad.append((boxes[i][0], boxes[j][0]))
    return bad


def tag(ax, letter):
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-30, 6), textcoords='offset points',
                fontsize=13, fontweight='bold', ha='left', va='bottom', color=INK)


# ------------------------------------------------------------------ Figure 1: flow
def flow_steps():
    """Main boxes and exclusion boxes (STROBE/RECORD style); every count comes from cohort_flow.json or the cohorts."""
    f = json.load(open(f"{OUT}/cohort_flow.json"))
    m, v = f['mover'], f['vitaldb']
    mc = pd.read_parquet(f"{OUT}/cohort_mover.parquet")
    vc = pd.read_parquet(f"{OUT}/cohort_vitaldb.parquet")
    n = lambda x: f"{int(x):,}"
    L = lambda *parts: "\n".join(parts)
    mover = [
        (L(f"{n(m['cases'])}", "anaesthetic cases"),
         L(f"{n(m['cases'] - m['adult_general'])} excluded:", "age <18 years, not general", "anaesthesia or sex not recorded")),
        (L(f"{n(m['adult_general'])} adults,", "general anaesthesia"),
         L(f"{n(m['excl_cardiac'] + m['excl_intracranial'] + m['excl_obstetric'])} excluded:",
           f"cardiac {n(m['excl_cardiac'])}, intracranial {n(m['excl_intracranial'])},",
           f"pregnancy-related {n(m['excl_obstetric'])}")),
        (L(f"{n(m['non_cardiac'])}", "eligible procedures"), L(f"{n(m['non_cardiac'] - m['anes_ge_60min'])} excluded:", "anaesthesia <60 min")),
        (L(f"{n(m['anes_ge_60min'])}", "anaesthesia ≥60 min"),
         L(f"{n(m['anes_ge_60min'] - m['first_case_per_patient'])} excluded:", "repeat operations")),
        (L(f"{n(m['first_case_per_patient'])} first", "operation per patient"),
         L(f"{n(m['first_case_per_patient'] - m['analysable'])} excluded:", "too few maintenance vital signs")),
        (L(f"{n(m['analysable'])}", "analysable patients"),
         L(f"{n(m['analysable'] - m['reference_asa12'])} excluded:", f"ASA III-VI {n((mc.asa >= 3).sum())}",
           f"ASA not recorded {n(mc.asa.isna().sum())}")),
        (L(f"{n(m['reference_asa12'])} ASA I-II:", "reference cohort"), None)]
    emerg = int(((vc.asa <= 2) & (vc.emop == 1)).sum())
    vdb = [
        (L(f"{n(v['cases'])}", "anaesthetic cases"),
         L(f"{n(v['cases'] - v['adult_general'])} excluded:", "age <18 years, not general", "anaesthesia or no monitor data")),
        (L(f"{n(v['adult_general'])} adults,", "general anaesthesia"), L(f"{n(v['excl_obstetric'])} excluded:", "caesarean section")),
        (L(f"{n(v['non_cardiac'])}", "eligible procedures"), L(f"{n(v['non_cardiac'] - v['anes_ge_60min'])} excluded:", "anaesthesia <60 min")),
        (L(f"{n(v['anes_ge_60min'])}", "anaesthesia ≥60 min"),
         L(f"{n(v['anes_ge_60min'] - v['first_case_per_subject'])} excluded:", "repeat operations")),
        (L(f"{n(v['first_case_per_subject'])} first operation", "per patient"),
         L(f"{n(v['first_case_per_subject'] - v['reference_asa12_elective'])} excluded:",
           f"ASA III-VI {n((vc.asa >= 3).sum())}, emergency {n(emerg)}", f"ASA not recorded {n(vc.asa.isna().sum())}")),
        (L(f"{n(v['reference_asa12_elective'])} ASA I-II,", "elective: validation"), None)]
    return mover, vdb


def fig_flow():
    mover, vdb = flow_steps()
    fig = plt.figure(figsize=(W, 6.0))
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    boxes_ = []

    def box(x, y, text, w, h, bold=False, size=6.8):
        pa = ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='square,pad=0', fc='white',
                                         ec=INK2, lw=0.7))
        boxes_.append((pa, ax.text(x, y, text, ha='center', va='center', fontsize=size, linespacing=1.15,
                                   fontweight='bold' if bold else 'normal')))

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.6, color=INK2,
                                                                      mutation_scale=6))
    MW, EW = 20, 23   # main and exclusion box widths; columns leave a margin at both figure edges
    for (xm, xe, head, steps) in ((12.0, 36.0, 'MOVER (UCI Medical Center)', mover),
                                  (61.0, 85.5, 'VitalDB (SNUH, Seoul)', vdb)):
        ax.text((xm + xe) / 2, 98.5, head, ha='center', va='center', fontsize=8.5, fontweight='bold')
        ys = np.linspace(91, 6, len(steps))
        step = ys[0] - ys[1]
        for i, (t, e) in enumerate(steps):
            box(xm, ys[i], t, MW, 6.6, bold=e is None, size=6.4)
            if i < len(steps) - 1:
                arrow(xm, ys[i] - 3.3, xm, ys[i + 1] + 3.3)
                ym = ys[i] - step / 2
                box(xe, ym, e, EW, 7.8, size=6.0)
                ax.plot([xm, xe - EW / 2], [ym, ym], color=INK2, lw=0.6)
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    ext = ax.get_window_extent(r)
    for pa, _ in boxes_:
        a = pa.get_window_extent(r)
        for qa, _ in boxes_:
            b = qa.get_window_extent(r)
            assert qa is pa or min(a.x1, b.x1) <= max(a.x0, b.x0) or min(a.y1, b.y1) <= max(a.y0, b.y0), "flow boxes overlap"
        assert a.x0 - ext.x0 > 0.01 * ext.width and ext.x1 - a.x1 > 0.01 * ext.width, "flow box touches the figure edge"
    for pa, t in boxes_:
        a, b = pa.get_window_extent(r), t.get_window_extent(r)
        assert b.x0 > a.x0 + 2 and b.x1 < a.x1 - 2 and b.y0 > a.y0 and b.y1 < a.y1, \
            f"flow text spills out of its box: {t.get_text()}"
    save(fig, 'figure1')


# ------------------------------------------------------------------ Figures 2-3: centile curves
def centile_panel(ax, var, sex, tab, ref):
    t = tab[(tab['var'] == var) & (tab.sex == sex)]
    c = SEXCOL[sex]
    ax.fill_between(t.age, t.P10, t.P90, color=c, alpha=0.13, lw=0)
    for q, ls, lw in (('P3', ':', 0.9), ('P10', '--', 0.9), ('P50', '-', 1.6), ('P90', '--', 0.9), ('P97', ':', 0.9)):
        ax.plot(t.age, t[q], color=c, ls=ls, lw=lw)
    # observed centiles in 5-year bins, to show the fit
    g = ref[ref.sex == sex][['age', var]].dropna()
    g = g.assign(b=(np.minimum(g.age, 89) // 5) * 5 + 2.5)
    e = g.groupby('b')[var].quantile([0.03, 0.5, 0.97]).unstack()
    for q in e.columns:
        ax.plot(e.index, e[q], 'o', ms=2.0, color=INK, alpha=0.55, mew=0)
    ax.set_xlim(18, 90); ax.set_xticks([20, 40, 60, 80])
    x = t.age.max()
    for q in ('P3', 'P50', 'P97'):
        ax.text(x + 1, t[q].iloc[-1], q, fontsize=7, color=c, va='center')


def fig_curves(vars_, name, ylims):
    tab = pd.read_csv(f"{OUT}/norms_table.csv")
    ref = pd.read_parquet(f"{OUT}/cohort_mover.parquet").query('ref')
    fig, axs = plt.subplots(2, 3, figsize=(W, 4.9), sharex=True)
    letters = iter('ABCDEF')
    for i, sex in enumerate(('Female', 'Male')):
        for j, v in enumerate(vars_):
            ax = axs[i, j]
            centile_panel(ax, v, sex, tab, ref)
            ax.set_ylim(*ylims[v])
            if v == 'nibp_map':
                ax.axhline(65, color=INK2, lw=0.6, ls=(0, (1, 2)))
                ax.text(89, 64.2, '65 mm Hg', fontsize=7, color=INK2, va='top', ha='right')
            if i == 0: ax.set_title(TITLE[v], fontsize=9, pad=6)
            if j == 0: ax.set_ylabel(f"{sex}\n{LAB[v]}", fontsize=8.5)
            else: ax.set_ylabel(LAB[v], fontsize=8.5)
            if i == 1: ax.set_xlabel('Age (years)')
            tag(ax, next(letters))
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    save(fig, name)


# ------------------------------------------------------------------ Figure 4: VitalDB calibration
DEPCOL = {'General surgery': '#9aa5b1', 'Thoracic surgery': '#eb6834', 'Gynecology': '#1baf7a', 'Urology': '#eda100'}
DEPLAB = {'General surgery': 'General surgery', 'Thoracic surgery': 'Thoracic surgery', 'Gynecology': 'Gynaecology',
          'Urology': 'Urology'}


def fig_calibration():
    d = pd.read_csv(f"{OUT}/validation.csv")
    V = ['nibp_map', 'nibp_sbp', 'nibp_dbp', 'hr', 'etco2', 'temp_c']
    Q = [3, 10, 25, 50, 75, 90, 97]
    fig, axs = plt.subplots(2, 3, figsize=(W, 5.0), sharex=True, sharey=True)
    for ax, v, L in zip(axs.flat, V, 'ABCDEF'):
        ax.plot([0, 100], [0, 100], color=INK2, lw=0.6, ls='--')
        for dep, c in DEPCOL.items():
            g = d[(d['var'] == v) & (d.stratum == 'Department') & (d.level == dep)]
            ax.plot(Q, [g[f'below_P{q}'].iloc[0] for q in Q], color=c, lw=0.8, alpha=0.95, label=DEPLAB[dep])
        a = d[(d['var'] == v) & (d.stratum == 'All')].iloc[0]
        y = [a[f'below_P{q}'] for q in Q]
        lo = [a[f'below_P{q}'] - a[f'below_P{q}_lo'] for q in Q]
        hi = [a[f'below_P{q}_hi'] - a[f'below_P{q}'] for q in Q]
        ax.errorbar(Q, y, yerr=[lo, hi], fmt='o', ms=3, color=INK, lw=0.8, capsize=0, zorder=5,
                    label='All patients (95% CI)')
        ax.set_title(TITLE[v], fontsize=9, pad=6)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
        tag(ax, L)
    for ax in axs[1]: ax.set_xlabel('MOVER reference centile')
    for ax in axs[:, 0]: ax.set_ylabel('VitalDB patients below (%)')
    h, l = axs[0, 0].get_legend_handles_labels()
    h[-1] = plt.Line2D([], [], marker='o', ls='', ms=3, color=INK)
    l[-1] = 'All patients'
    fig.legend(h[::-1], l[::-1], loc='lower center', ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.4, h_pad=1.4, rect=(0, 0.05, 1, 1))
    save(fig, 'figure4')


# ------------------------------------------------------------------ Figure S1: primary vs no vasoactive drug
def fig_nodrug():
    a = pd.read_csv(f"{OUT}/norms_table.csv")
    b = pd.read_csv(f"{OUT}/norms_table_nodrug.csv")
    fig, axs = plt.subplots(2, 3, figsize=(W, 4.9), sharex=True)
    letters = iter('ABCDEF')
    ylim = {'nibp_map': (55, 125), 'nibp_sbp': (75, 190), 'hr': (45, 120)}
    for i, sex in enumerate(('Female', 'Male')):
        for j, v in enumerate(['nibp_map', 'nibp_sbp', 'hr']):
            ax = axs[i, j]
            for t, ls, lab in ((a, '-', 'All ASA I-II (primary)'), (b, '--', 'No vasoactive drug')):
                g = t[(t['var'] == v) & (t.sex == sex) & (t.age <= 85)]
                for q in ('P3', 'P50', 'P97'):
                    ax.plot(g.age, g[q], color=SEXCOL[sex], ls=ls, lw=1.4 if q == 'P50' else 0.9,
                            label=lab if q == 'P50' else None)
            g = a[(a['var'] == v) & (a.sex == sex) & (a.age == 85)]
            for q in ('P3', 'P50', 'P97'):
                ax.text(86, g[q].iloc[0], q, fontsize=7, color=SEXCOL[sex], va='center')
            ax.set_ylim(*ylim[v]); ax.set_xlim(18, 90); ax.set_xticks([20, 40, 60, 80])
            if i == 0: ax.set_title(TITLE[v], fontsize=9, pad=6)
            ax.set_ylabel(f"{sex}\n{LAB[v]}" if j == 0 else LAB[v], fontsize=8.5)
            if i == 1: ax.set_xlabel('Age (years)')
            tag(ax, next(letters))
    axs[0, 0].legend(frameon=False, fontsize=7, loc='upper left')
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    save(fig, 'figureS1')


# ------------------------------------------------------------------ Figure S2: recalibration
def fig_recal():
    r = pd.read_csv(f"{OUT}/recalibration.csv", keep_default_na=False)
    V = ['nibp_map', 'nibp_sbp', 'nibp_dbp', 'hr', 'etco2', 'temp_c']
    Q = [3, 10, 50, 90, 97]
    sty = {'None': ('o', INK, 'MOVER centiles as published'), 'Shift': ('s', '#2a78d6', 'Shifted by the median offset'),
           'Shift and scale': ('^', '#eb6834', 'Shifted and rescaled')}
    fig, axs = plt.subplots(2, 3, figsize=(W, 4.9), sharex=True, sharey=True)
    for ax, v, L in zip(axs.flat, V, 'ABCDEF'):
        ax.plot([0, 100], [0, 100], color=INK2, lw=0.6, ls='--')
        for mth, (mk, c, lab) in sty.items():
            g = r[(r['var'] == v) & (r.method == mth)].iloc[0]
            ax.plot(Q, [g[f'below_P{q}'] for q in Q], marker=mk, ms=3.2, color=c, lw=0.8, label=lab)
        ax.set_title(TITLE[v], fontsize=9, pad=6); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
        tag(ax, L)
    for ax in axs[1]: ax.set_xlabel('Reference centile')
    for ax in axs[:, 0]: ax.set_ylabel('Patients below (%)')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.4, h_pad=1.4, rect=(0, 0.05, 1, 1))
    save(fig, 'figureS2')


if __name__ == '__main__':
    fig_flow()
    fig_curves(['nibp_map', 'nibp_sbp', 'nibp_dbp'], 'figure2',
               {'nibp_map': (55, 120), 'nibp_sbp': (75, 175), 'nibp_dbp': (40, 90)})
    fig_curves(['hr', 'etco2', 'temp_c'], 'figure3', {'hr': (45, 120), 'etco2': (20, 55), 'temp_c': (34, 38)})
    fig_calibration()
    fig_nodrug()
    fig_recal()
    print('figures done')
