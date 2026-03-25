from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from rayleigh import rayleigh_window_group_df

WINDOWS = ((1, 4), (4, 7))
WINDOW_LABELS = {(1, 4): 'fixation_window', (4, 7): 'cue_window'}
WHICH_HEMI = tuple(product(('start', 'end'), (0, 1)))
GROUP_META = {
    'ctrl': {'sheet': 'Sheet1', 'id_col': 'ctrl', 'label': 'control'},
    'pat': {'sheet': 'Sheet2', 'id_col': 'pat', 'label': 'patient'},
}


def load_group_performance(xlsx_path: Path, group_code: str) -> pd.DataFrame:
    meta = GROUP_META[group_code]
    df = pd.read_excel(xlsx_path, sheet_name=meta['sheet'])
    cols = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame({
        'group': group_code,
        'group_label': meta['label'],
        'participant': pd.to_numeric(df[cols[meta['id_col']]], errors='coerce').astype('Int64'),
        'mean_perf': pd.to_numeric(df[cols['mean perf']], errors='coerce'),
    }).dropna(subset=['participant', 'mean_perf'])
    out['participant'] = out['participant'].astype(int)
    return out


def build_phase_sig_by_window(group_code: str, which: str, hemi: int) -> pd.DataFrame:
    rdf = rayleigh_window_group_df(group=group_code, hemi=hemi, which=which, windows=WINDOWS, final=False).copy()
    rdf['window'] = list(zip(rdf['window_start_s'], rdf['window_end_s']))
    rdf['window'] = rdf['window'].map(WINDOW_LABELS)
    rdf['phase_sig'] = rdf['p'] < 0.05
    return rdf[['participant', 'window', 'phase_sig']].assign(group=group_code, hemi=hemi, which=which)


def collapse_phase_sig(sig_by_window: pd.DataFrame) -> pd.DataFrame:
    return sig_by_window.groupby(['group', 'participant', 'which', 'hemi'], as_index=False)['phase_sig'].max()


def run_tests_collapsed(perf_df: pd.DataFrame, sig_collapsed: pd.DataFrame) -> pd.DataFrame:
    merged = perf_df.merge(sig_collapsed, on=['group', 'participant'], how='inner')
    rows = []
    for group_code, meta in GROUP_META.items():
        g = merged[merged['group'] == group_code]
        for which, hemi in WHICH_HEMI:
            sub = g[(g['which'] == which) & (g['hemi'] == hemi)]
            sig_vals = sub[sub['phase_sig']]['mean_perf'].to_numpy(dtype=float)
            nosig_vals = sub[~sub['phase_sig']]['mean_perf'].to_numpy(dtype=float)
            mean_sig = float(np.nanmean(sig_vals))
            mean_nonsig = float(np.nanmean(nosig_vals))
            t = stats.ttest_ind(sig_vals, nosig_vals, equal_var=False, nan_policy='omit')
            t_welch, p_welch = float(t.statistic), float(t.pvalue)
            rows.append({
                'group': meta['label'],
                'which': which,
                'hemi': hemi,
                'window': 'collapsed_fixation_plus_cue',
                'n_sig': int(sig_vals.size),
                'n_nonsig': int(nosig_vals.size),
                'mean_perf_sig': mean_sig,
                'mean_perf_nonsig': mean_nonsig,
                'lower_is_better_direction': ('nonsig_better', 'sig_better')[mean_sig < mean_nonsig],
                't_welch': t_welch,
                'p_welch': p_welch,
            })
    return pd.DataFrame(rows)


def main():
    xlsx = Path('/Users/anji/Library/Mobile Documents/com~apple~CloudDocs/Downloads/mean_perf.xlsx')
    out_dir = Path('analysis_outputs/performance_phase')
    out_dir.mkdir(parents=True, exist_ok=True)

    perf = pd.concat([load_group_performance(xlsx, g) for g in GROUP_META], ignore_index=True)

    cache_by_window = out_dir / 'group_phase_sig_by_window.csv'
    if cache_by_window.exists():
        sig_by_window = pd.read_csv(cache_by_window)
    else:
        sig_by_window = pd.concat(
            [build_phase_sig_by_window(group_code=g, which=w, hemi=h) for g in GROUP_META for w, h in WHICH_HEMI],
            ignore_index=True,
        )
        sig_by_window.to_csv(cache_by_window, index=False)

    sig_collapsed = collapse_phase_sig(sig_by_window)
    res_collapsed = run_tests_collapsed(perf, sig_collapsed)

    perf.to_csv(out_dir / 'group_performance_clean.csv', index=False)
    sig_collapsed.to_csv(out_dir / 'group_phase_sig_collapsed.csv', index=False)
    res_collapsed.to_csv(out_dir / 'group_perf_ttests_collapsed_window_hemi.csv', index=False)

    print(res_collapsed.to_string(index=False))
    print(f"\nSaved to: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
