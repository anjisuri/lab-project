from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from rayleigh import rayleigh_window_group_df

WINDOW_LABELS = {
    (1, 4): 'fixation_window',
    (4, 7): 'cue_window',
}

GROUP_META = {
    'ctrl': {'sheet': 'Sheet1', 'id_col': 'ctrl', 'label': 'control'},
    'pat': {'sheet': 'Sheet2', 'id_col': 'pat', 'label': 'patient'},
}


def load_group_performance(xlsx_path: Path, group_code: str) -> pd.DataFrame:
    meta = GROUP_META[group_code]
    df = pd.read_excel(xlsx_path, sheet_name=meta['sheet'])
    cols = {c.lower().strip(): c for c in df.columns}

    if meta['id_col'] not in cols or 'mean perf' not in cols:
        raise ValueError(f"{meta['sheet']} must contain columns: {meta['id_col']}, mean perf")

    out = pd.DataFrame(
        {
            'group': group_code,
            'group_label': meta['label'],
            'participant': pd.to_numeric(df[cols[meta['id_col']]], errors='coerce').astype('Int64'),
            'mean_perf': pd.to_numeric(df[cols['mean perf']], errors='coerce'),
        }
    )
    out = out.dropna(subset=['participant', 'mean_perf']).copy()
    out['participant'] = out['participant'].astype(int)
    return out


def build_phase_sig_by_window(group_code: str, which: str, hemi: int) -> pd.DataFrame:
    rdf = rayleigh_window_group_df(
        group=group_code,
        hemi=hemi,
        which=which,
        windows=((1, 4), (4, 7)),
        final=False,
    )
    if rdf.empty:
        return pd.DataFrame(columns=['group', 'participant', 'window', 'hemi', 'which', 'phase_sig'])

    rdf = rdf.copy()
    rdf['window'] = list(zip(rdf['window_start_s'], rdf['window_end_s']))
    rdf['window'] = rdf['window'].map(WINDOW_LABELS)
    rdf['phase_sig'] = rdf['p'] < 0.05

    return rdf[['participant', 'window', 'phase_sig']].assign(group=group_code, hemi=hemi, which=which)


def collapse_phase_sig(sig_by_window: pd.DataFrame) -> pd.DataFrame:
    # Collapsed across fixation+cue: phase_sig=True if significant in either window.
    if sig_by_window.empty:
        return pd.DataFrame(columns=['group', 'participant', 'which', 'hemi', 'phase_sig'])

    out = (
        sig_by_window.groupby(['group', 'participant', 'which', 'hemi'], as_index=False)['phase_sig']
        .max()
    )
    return out


def run_tests_collapsed(perf_df: pd.DataFrame, sig_collapsed: pd.DataFrame) -> pd.DataFrame:
    merged = perf_df.merge(sig_collapsed, on=['group', 'participant'], how='inner')
    rows = []

    for group_code, group_label in [('ctrl', 'control'), ('pat', 'patient')]:
        g = merged[merged['group'] == group_code]
        for which in ['start', 'end']:
            for hemi in [0, 1]:
                sub = g[(g['which'] == which) & (g['hemi'] == hemi)]
                sig_vals = sub[sub['phase_sig']]['mean_perf'].to_numpy(dtype=float)
                nosig_vals = sub[~sub['phase_sig']]['mean_perf'].to_numpy(dtype=float)

                row = {
                    'group': group_label,
                    'which': which,
                    'hemi': hemi,
                    'window': 'collapsed_fixation_plus_cue',
                    'n_sig': int(sig_vals.size),
                    'n_nonsig': int(nosig_vals.size),
                    'mean_perf_sig': float(np.nanmean(sig_vals)) if sig_vals.size else np.nan,
                    'mean_perf_nonsig': float(np.nanmean(nosig_vals)) if nosig_vals.size else np.nan,
                    'lower_is_better_direction': (
                        'sig_better'
                        if sig_vals.size and nosig_vals.size and np.nanmean(sig_vals) < np.nanmean(nosig_vals)
                        else ('nonsig_better' if sig_vals.size and nosig_vals.size else np.nan)
                    ),
                }

                if sig_vals.size >= 2 and nosig_vals.size >= 2:
                    t = stats.ttest_ind(sig_vals, nosig_vals, equal_var=False, nan_policy='omit')
                    row.update({'t_welch': float(t.statistic), 'p_welch': float(t.pvalue)})
                else:
                    row.update({'t_welch': np.nan, 'p_welch': np.nan})

                rows.append(row)

    return pd.DataFrame(rows)


def main():
    xlsx = Path('/Users/anji/Library/Mobile Documents/com~apple~CloudDocs/Downloads/mean_perf.xlsx')
    out_dir = Path('analysis_outputs/performance_phase')
    out_dir.mkdir(parents=True, exist_ok=True)

    perf_parts = [load_group_performance(xlsx, 'ctrl'), load_group_performance(xlsx, 'pat')]
    perf = pd.concat(perf_parts, ignore_index=True)

    # Use cached by-window sig if present, otherwise compute.
    cache_by_window = out_dir / 'group_phase_sig_by_window.csv'
    if cache_by_window.exists():
        sig_by_window = pd.read_csv(cache_by_window)
    else:
        sig_parts = []
        for group_code in ['ctrl', 'pat']:
            for which in ['start', 'end']:
                for hemi in [0, 1]:
                    sig_parts.append(build_phase_sig_by_window(group_code=group_code, which=which, hemi=hemi))
        sig_by_window = pd.concat(sig_parts, ignore_index=True)
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
