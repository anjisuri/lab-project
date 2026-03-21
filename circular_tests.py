import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# Avoid Matplotlib cache warnings from indirect imports in this pipeline.
MPL_CACHE = Path('analysis_outputs') / '.mpl_cache'
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPL_CACHE.resolve()))

from config import get_common_participant_ids
from rayleigh import phase_dist_window, rayleigh_test


WINDOWS = ((1, 4), (4, 7))  # fixation, cue


def circular_mean(phases):
    phases = np.asarray(phases, dtype=float)
    phases = phases[np.isfinite(phases)]
    if phases.size == 0:
        return np.nan
    return float(np.angle(np.mean(np.exp(1j * phases))))


def circular_distance(a, b):
    """Signed circular distance a-b in [-pi, pi]."""
    return float(np.angle(np.exp(1j * (a - b))))


def v_test_against_zero(angles):
    """One-sample circular V-test toward 0 radians."""
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    n = angles.size
    if n == 0:
        return {"n": 0, "mu": np.nan, "r_bar": np.nan, "v": np.nan, "u": np.nan, "p": np.nan}

    c = np.sum(np.cos(angles))
    s = np.sum(np.sin(angles))
    mu = np.arctan2(s, c)
    r_bar = np.sqrt(c * c + s * s) / n

    # V-test with specified mean direction mu0 = 0
    v = r_bar * np.cos(mu)
    u = v * np.sqrt(2.0 * n)
    p = 1.0 - norm.cdf(u)

    return {
        "n": int(n),
        "mu": float(mu),
        "r_bar": float(r_bar),
        "v": float(v),
        "u": float(u),
        "p": float(np.clip(p, 0.0, 1.0)),
    }


def run_phase_difference_test(which='start', hemi=0):
    rows = []

    for group in ('ctrl', 'pat'):
        ids = get_common_participant_ids(group, windows=WINDOWS)
        for pid in ids:
            fix_phases = phase_dist_window(
                pid,
                group=group,
                hemi=hemi,
                which=which,
                window=WINDOWS[0],
                final=False,
            )
            cue_phases = phase_dist_window(
                pid,
                group=group,
                hemi=hemi,
                which=which,
                window=WINDOWS[1],
                final=False,
            )

            mu_fix = circular_mean(fix_phases)
            mu_cue = circular_mean(cue_phases)
            if np.isfinite(mu_fix) and np.isfinite(mu_cue):
                delta = circular_distance(mu_cue, mu_fix)  # cue - fixation
                rows.append(
                    {
                        'hemi': int(hemi),
                        'group': group,
                        'participant': int(pid),
                        'mu_fix': float(mu_fix),
                        'mu_cue': float(mu_cue),
                        'delta_cue_minus_fix': float(delta),
                        'abs_delta': float(abs(delta)),
                    }
                )

    per_subject = pd.DataFrame(rows)

    test_rows = []
    for grp in ('ctrl', 'pat', 'all'):
        sub = per_subject if grp == 'all' else per_subject[per_subject['group'] == grp]
        deltas = sub['delta_cue_minus_fix'].to_numpy(dtype=float) if not sub.empty else np.array([])

        vtest = v_test_against_zero(deltas)
        z, p_rayleigh = rayleigh_test(deltas) if vtest['n'] > 0 else (np.nan, np.nan)

        test_rows.append(
            {
                'hemi': int(hemi),
                'group': grp,
                'n': int(vtest['n']),
                'mean_delta': float(vtest['mu']),
                'r_bar': float(vtest['r_bar']),
                'v': float(vtest['v']),
                'u': float(vtest['u']),
                'p_vtest_against_0': float(vtest['p']),
                'rayleigh_z': float(z),
                'rayleigh_p': float(p_rayleigh),
            }
        )

    return per_subject, pd.DataFrame(test_rows)


def save_and_print(which='start', hemis=(0, 1)):
    out_dir = Path('analysis_outputs/circular')
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_subject = []
    combined_tests = []

    for hemi in hemis:
        per_subject, tests_df = run_phase_difference_test(which=which, hemi=hemi)

        per_subject_path = out_dir / f'phase_delta_participant_{which}_h{hemi}.csv'
        tests_path = out_dir / f'phase_delta_tests_{which}_h{hemi}.csv'

        per_subject.to_csv(per_subject_path, index=False)
        tests_df.to_csv(tests_path, index=False)

        combined_subject.append(per_subject)
        combined_tests.append(tests_df)

        print(f'\n=== Hemisphere {hemi} | Per-participant circular mean difference (cue - fixation) ===')
        print(per_subject.to_string(index=False) if not per_subject.empty else 'No participant rows')

        print(f'\n=== Hemisphere {hemi} | Circular test: is difference centered at 0? ===')
        print(tests_df.to_string(index=False))

        print(f'\nSaved: {per_subject_path}')
        print(f'Saved: {tests_path}')

    subject_all = pd.concat(combined_subject, ignore_index=True) if combined_subject else pd.DataFrame()
    tests_all = pd.concat(combined_tests, ignore_index=True) if combined_tests else pd.DataFrame()

    subject_all_path = out_dir / f'phase_delta_participant_{which}_both_hemi.csv'
    tests_all_path = out_dir / f'phase_delta_tests_{which}_both_hemi.csv'
    subject_all.to_csv(subject_all_path, index=False)
    tests_all.to_csv(tests_all_path, index=False)

    print(f'\nSaved combined: {subject_all_path}')
    print(f'Saved combined: {tests_all_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-participant circular mean phase differences (cue-fix) and test vs 0 using a circular test.'
    )
    parser.add_argument('--which', choices=['start', 'end'], default='start', help='start=saccade onset, end=fixation onset')
    parser.add_argument('--hemi', choices=['0', '1', 'both'], default='both', help='hemisphere: 0 (left), 1 (right), or both')
    args = parser.parse_args()

    if args.hemi == 'both':
        hemis = (0, 1)
    else:
        hemis = (int(args.hemi),)

    save_and_print(which=args.which, hemis=hemis)


if __name__ == '__main__':
    main()
