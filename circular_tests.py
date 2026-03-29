import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pycircstat2.descriptive import circ_mean_ci
from pycircstat2.hypothesis import one_sample_test


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
    # circular distance a-b in [-pi, pi]
    return float(np.angle(np.exp(1j * (a - b))))


def one_sample_test_against_zero(angles):
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    n = angles.size
    if n == 0:
        return {"n": 0, "mu": np.nan, "r_bar": np.nan, "reject": np.nan, "ci_lb": np.nan, "ci_ub": np.nan}

    c = np.sum(np.cos(angles))
    s = np.sum(np.sin(angles))
    mu = np.arctan2(s, c)
    r_bar = np.sqrt(c * c + s * s) / n

    try:
        try:
            one_sample_res = one_sample_test(angle=0.0, alpha=angles)
        except ValueError:
            ci_method = "bootstrap" if n < 25 else "dispersion"
            lb, ub = circ_mean_ci(alpha=angles, method=ci_method)
            one_sample_res = one_sample_test(angle=0.0, lb=lb, ub=ub)
        ci = getattr(one_sample_res, "ci", (np.nan, np.nan))
        reject = getattr(one_sample_res, "reject", np.nan)
        status = "ok"
    except ValueError:
        # extremely low resultant length: mean direction/CI is not defined.
        ci = (np.nan, np.nan)
        reject = np.nan
        status = "undefined_mean_direction"

    return {
        "n": int(n),
        "mu": float(mu),
        "r_bar": float(r_bar),
        "reject": bool(reject) if not pd.isna(reject) else np.nan,
        "ci_lb": float(ci[0]),
        "ci_ub": float(ci[1]),
        "status": status,
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

        one_sample = one_sample_test_against_zero(deltas)
        z, p_rayleigh = rayleigh_test(deltas) if one_sample['n'] > 0 else (np.nan, np.nan)

        test_rows.append(
            {
                'hemi': int(hemi),
                'group': grp,
                'n': int(one_sample['n']),
                'mean_delta': float(one_sample['mu']),
                'r_bar': float(one_sample['r_bar']),
                'one_sample_reject_mean_eq_0': one_sample['reject'],
                'one_sample_ci_lb': float(one_sample['ci_lb']),
                'one_sample_ci_ub': float(one_sample['ci_ub']),
                'one_sample_status': one_sample['status'],
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

        print(f'\n=== Hemisphere {hemi} | one_sample_test: is mean difference equal to 0? ===')
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
        description='Compute per-participant circular mean phase differences (cue-fix) and test mean=0 with pycircstat2 one_sample_test.'
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
