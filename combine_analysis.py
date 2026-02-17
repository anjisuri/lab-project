import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Helvetica'


def sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def split(w):
    d = w.copy()
    d['mean'] = (d.fixation + d.cue) / 2
    d['diff'] = d.cue - d.fixation
    c = d[d.group == 'control']
    p = d[d.group == 'patient']

    tg = stats.ttest_ind(c['mean'], p['mean'], equal_var=False)
    tc = stats.ttest_1samp(d['diff'], 0.0)
    ti = stats.ttest_ind(c['diff'], p['diff'], equal_var=False)

    n1, n2 = len(c), len(p)
    s1 = np.var(c['mean'], ddof=1)
    s2 = np.var(p['mean'], ddof=1)
    dfg = (s1 / n1 + s2 / n2) ** 2 / ((s1**2) / (n1**2 * (n1 - 1)) + (s2**2) / (n2**2 * (n2 - 1)))

    s1 = np.var(c['diff'], ddof=1)
    s2 = np.var(p['diff'], ddof=1)
    dfi = (s1 / n1 + s2 / n2) ** 2 / ((s1**2) / (n1**2 * (n1 - 1)) + (s2**2) / (n2**2 * (n2 - 1)))

    cells = []
    for g in ['control', 'patient']:
        sub = d[d.group == g]
        for cond in ['fixation', 'cue']:
            vals = sub[cond].to_numpy()
            vals = vals[np.isfinite(vals)]
            cells.append(
                {
                    'group': g,
                    'condition': cond,
                    'mean': float(np.mean(vals)),
                    'sem': float(sem(vals)),
                    'n': int(len(vals)),
                }
            )

    return {
        'n_control': n1,
        'n_patient': n2,
        'group_F': float(tg.statistic**2),
        'group_df2': float(dfg),
        'group_p': float(tg.pvalue),
        'condition_F': float(tc.statistic**2),
        'condition_df2': float(len(d) - 1),
        'condition_p': float(tc.pvalue),
        'interaction_F': float(ti.statistic**2),
        'interaction_df2': float(dfi),
        'interaction_p': float(ti.pvalue),
        'cells': pd.DataFrame(cells),
    }


def make_wide(ctrl_csv, pat_csv):
    c = pd.read_csv(ctrl_csv)
    p = pd.read_csv(pat_csv)
    c['group'] = 'control'
    p['group'] = 'patient'
    c['subject'] = c['participant'].astype(int).map(lambda x: f'C{x:02d}')
    p['subject'] = p['participant'].astype(int).map(lambda x: f'P{x:02d}')

    d = pd.concat([c, p], ignore_index=True)
    d['condition'] = list(zip(d['window_start_s'], d['window_end_s']))
    d['condition'] = d['condition'].map({(1, 4): 'fixation', (4, 7): 'cue'})

    w = d.pivot_table(index=['subject', 'group'], columns='condition', values='r', aggfunc='mean').dropna().reset_index()
    return w[['subject', 'group', 'fixation', 'cue']]


def plot_cells(cells, title, ylabel, fname):
    groups = ['control', 'patient']
    conds = ['fixation', 'cue']
    x = np.arange(2) * 0.45
    width = 0.12

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for j, cond in enumerate(conds):
        means, errs = [], []
        for g in groups:
            r = cells[(cells.group == g) & (cells.condition == cond)].iloc[0]
            means.append(r['mean'])
            errs.append(r['sem'])
        pos = x + (j - 0.5) * width
        ax.bar(pos, means, width=width, label=cond.capitalize(), alpha=0.9)
        ax.errorbar(pos, means, yerr=errs, fmt='none', capsize=4, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(['Controls', 'Patients'])
    ax.set_xlim(x[0] - 0.22, x[-1] + 0.22)
    ax.margins(x=0.01)
    ax.set_ylabel(ylabel, rotation=90, labelpad=14, va='center')
    ax.set_title(title, loc='center', pad=10)
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.02), frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close(fig)


rows = []
mapping = [
    ('rates_wide.csv', 'Saccade Frequency', 'Frequency (Hz)', 'analysis_outputs/saccade_frequency_bar.png', 'saccade_frequency'),
    ('durations_wide.csv', 'Saccade Duration', 'Duration (ms)', 'analysis_outputs/saccade_duration_bar.png', 'saccade_duration'),
    ('fixation_rates_wide.csv', 'Fixation Frequency', 'Frequency (Hz)', 'analysis_outputs/fixation_frequency_bar.png', 'fixation_frequency'),
    ('fixation_durations_wide.csv', 'Fixation Duration', 'Duration (ms)', 'analysis_outputs/fixation_duration_bar.png', 'fixation_duration'),
]

for csv, title, ylabel, png, name in mapping:
    w = pd.read_csv('analysis_outputs/' + csv)
    r = split(w)
    r['cells'].to_csv(f'analysis_outputs/{name}_cells.csv', index=False)
    plot_cells(r['cells'], title, ylabel, png)
    for ef in ['group', 'condition', 'interaction']:
        rows.append(
            {
                'analysis': name,
                'effect': ef,
                'F': r[f'{ef}_F'],
                'df1': 1,
                'df2': r[f'{ef}_df2'],
                'p': r[f'{ef}_p'],
                'n_control': r['n_control'],
                'n_patient': r['n_patient'],
            }
        )

w_start = make_wide('analysis_outputs/phase_start_ctrl_long.csv', 'analysis_outputs/phase_start_patient_long.csv')
w_end = make_wide('analysis_outputs/phase_end_ctrl_long.csv', 'analysis_outputs/phase_end_patient_long.csv')
w_start.to_csv('analysis_outputs/phase_saccade_onset_wide.csv', index=False)
w_end.to_csv('analysis_outputs/phase_fixation_onset_wide.csv', index=False)

r = split(w_start)
r['cells'].to_csv('analysis_outputs/phase_saccade_onset_cells.csv', index=False)
plot_cells(r['cells'], 'Phase Locking: Saccade Onset', 'Resultant Vector Length (r)', 'analysis_outputs/phase_saccade_onset_bar.png')
for ef in ['group', 'condition', 'interaction']:
    rows.append(
        {
            'analysis': 'phase_saccade_onset',
            'effect': ef,
            'F': r[f'{ef}_F'],
            'df1': 1,
            'df2': r[f'{ef}_df2'],
            'p': r[f'{ef}_p'],
            'n_control': r['n_control'],
            'n_patient': r['n_patient'],
        }
    )

r = split(w_end)
r['cells'].to_csv('analysis_outputs/phase_fixation_onset_cells.csv', index=False)
plot_cells(r['cells'], 'Phase Locking: Fixation Onset', 'Resultant Vector Length (r)', 'analysis_outputs/phase_fixation_onset_bar.png')
for ef in ['group', 'condition', 'interaction']:
    rows.append(
        {
            'analysis': 'phase_fixation_onset',
            'effect': ef,
            'F': r[f'{ef}_F'],
            'df1': 1,
            'df2': r[f'{ef}_df2'],
            'p': r[f'{ef}_p'],
            'n_control': r['n_control'],
            'n_patient': r['n_patient'],
        }
    )

out = pd.DataFrame(rows)
out.to_csv('analysis_outputs/all_anova_summary.csv', index=False)
print(out.to_string(index=False))
