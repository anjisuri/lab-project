import numpy as np
import pandas as pd
from scipy import stats

import sf_con
import sf_pat
from meg_filtering import hilbert
from config import get_control_file, get_patient_file, get_common_participant_ids


def resultant_vector_length(phases):
    phases = np.asarray(phases)
    phases = phases[~np.isnan(phases)]
    return np.abs(np.mean(np.exp(1j * phases)))


def rayleigh_test(phases):
    phases = np.asarray(phases)
    phases = phases[~np.isnan(phases)]
    n = phases.size # number of phase samples
    r = np.abs(np.mean(np.exp(1j * phases))) # resultant vector length
    z = n * r * r # larger = more clustering
    if n < 50:
        p = np.exp(-z) * (
            1 + (2 * z - z ** 2) / (4 * n)
            - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2)
        ) # small p - significant phase clustering
    else:
        p = np.exp(-z)
    return z, float(np.clip(p, 0.0, 1.0))


def saccade_phases_in_window(participant, trial, group="ctrl", hemi=0, fs=200, which="start", window=None, final=False):
    eye = sf_con if group == "ctrl" else sf_pat
    qc, valid_stats, _ = eye.trial(
        participant,
        trial,
        show_plot=False,
        show_stats=False,
        final=final,
    )
    out = hilbert(participant, trial, group=group, show_plots=False, fs=fs)
    phase = out["phase"]

    starts = valid_stats["start_idx"].to_numpy()
    ends = valid_stats["end_idx"].to_numpy()

    if window is not None:
        start_s, end_s = window
        starts = starts[(starts / fs >= start_s) & (starts / fs <= end_s)]
        ends = ends[(ends / fs >= start_s) & (ends / fs <= end_s)]

    if which == "start":
        return phase[starts, hemi]
    if which == "end":
        return phase[ends, hemi]
    
    return np.concatenate([phase[starts, hemi], phase[ends, hemi]])


def phase_dist_window(participant, group="ctrl", hemi=0, fs=200, which="start", window=None, final=False):
    file_path = get_control_file(participant) if group == "ctrl" else get_patient_file(participant)
    data = np.load(file_path)
    n_trials = data.shape[2]

    phases = []
    for trial in range(1, n_trials + 1):
        trial_phases = saccade_phases_in_window(
            participant,
            trial,
            group=group,
            hemi=hemi,
            fs=fs,
            which=which,
            window=window,
            final=final,
        )
        phases.extend(trial_phases)

    return np.asarray(phases)


def rayleigh_window_stats(participant, group="ctrl", hemi=0, fs=200, which="start", window=None, final=False):
    phases = phase_dist_window(
        participant,
        group=group,
        hemi=hemi,
        fs=fs,
        which=which,
        window=window,
        final=final,
    )
    r = resultant_vector_length(phases)
    z, p = rayleigh_test(phases)
    return {
        "participant": participant,
        "n": int(phases.size),
        "r": r,
        "z": z,
        "p": p,
    }


def rayleigh_window_group_stats(group="ctrl", hemi=0, fs=200, which="start", windows=((1, 4), (4, 7)), final=False):
    results = {window: [] for window in windows}
    participants = get_common_participant_ids(group=group, windows=windows)
    for participant in participants:
        for window in windows:
            stats_row = rayleigh_window_stats(
                participant,
                group=group,
                hemi=hemi,
                fs=fs,
                which=which,
                window=window,
                final=final,
            )
            if not np.isnan(stats_row["r"]):
                results[window].append(stats_row)
    return results


def compare_resultant_lengths(
    group="ctrl",
    hemi=0,
    fs=200,
    which="start",
    windows=((1, 4), (4, 7)),
    paired=True,
    final=False,
):
    window_a, window_b = windows
    results = rayleigh_window_group_stats(
        group=group,
        hemi=hemi,
        fs=fs,
        which=which,
        windows=windows,
        final=final,
    )
    a_map = {row["participant"]: row["r"] for row in results[window_a]}
    b_map = {row["participant"]: row["r"] for row in results[window_b]}
    common = sorted(set(a_map) & set(b_map))
    a_vals = np.asarray([a_map[p] for p in common])
    b_vals = np.asarray([b_map[p] for p in common])
    if paired:
        test = stats.ttest_rel(a_vals, b_vals, nan_policy="omit")
    else:
        test = stats.ttest_ind(a_vals, b_vals, nan_policy="omit")
    return {
        "window_a": window_a,
        "window_b": window_b,
        "paired": paired,
        "n": int(len(common)),
        "t": float(test.statistic),
        "p": float(test.pvalue),
        "a_vals": a_vals,
        "b_vals": b_vals,
    }


def rayleigh_window_group_df(group="ctrl", hemi=0, fs=200, which="start", windows=((1, 4), (4, 7)), final=False):
    results = rayleigh_window_group_stats(
        group=group,
        hemi=hemi,
        fs=fs,
        which=which,
        windows=windows,
        final=final,
    )
    rows = []
    for window, stats_rows in results.items():
        start_s, end_s = window
        for row in stats_rows:
            rows.append(
                {
                    "group": group,
                    "window_start_s": start_s,
                    "window_end_s": end_s,
                    "participant": row["participant"],
                    "n": row["n"],
                    "r": row["r"],
                    "z": row["z"],
                    "p": row["p"],
                }
            )
    
    return pd.DataFrame(rows)

def rayleigh_paired_window_ttests(df, windows=((1, 4), (4, 7))):
    results = {}
    if df.empty:
        return results

    window_a, window_b = windows
    if window_a == window_b:
        return results

    for group in sorted(df["group"].unique()):
        sub = df[df["group"] == group]
        pivot = sub.pivot_table(
            index="participant",
            columns=["window_start_s", "window_end_s"],
            values="r",
            aggfunc="mean",
        )
        if window_a in pivot.columns and window_b in pivot.columns:
            a = pivot[window_a].to_numpy()
            b = pivot[window_b].to_numpy()
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() > 1:
                test = stats.ttest_rel(a[mask], b[mask], nan_policy="omit")
                results[group] = {
                    "window_a": window_a,
                    "window_b": window_b,
                    "n": int(mask.sum()),
                    "t": float(test.statistic),
                    "p": float(test.pvalue),
                }
    return results


def rayleigh_between_group_ttests(df, windows=((1, 4), (4, 7))):
    results = {}
    if df.empty:
        return results

    for window in windows:
        start_s, end_s = window
        a = df[
            (df["group"] == "ctrl")
            & (df["window_start_s"] == start_s)
            & (df["window_end_s"] == end_s)
        ]["r"].to_numpy()
        b = df[
            (df["group"] == "pat")
            & (df["window_start_s"] == start_s)
            & (df["window_end_s"] == end_s)
        ]["r"].to_numpy()
        if a.size > 1 and b.size > 1:
            test = stats.ttest_ind(a, b, nan_policy="omit", equal_var=False)
            results[window] = {
                "window": window,
                "n_ctrl": int(np.isfinite(a).sum()),
                "n_patient": int(np.isfinite(b).sum()),
                "t": float(test.statistic),
                "p": float(test.pvalue),
            }

    return results


def rayleigh_default_ttests(hemi=0, fs=200, which="start", windows=((1, 4), (4, 7)), final=False):
    df_ctrl = rayleigh_window_group_df(
        group="ctrl",
        hemi=hemi,
        fs=fs,
        which=which,
        windows=windows,
        final=final,
    )
    df_pat = rayleigh_window_group_df(
        group="pat",
        hemi=hemi,
        fs=fs,
        which=which,
        windows=windows,
        final=final,
    )
    df = pd.concat([df_ctrl, df_pat], ignore_index=True)
    return {
        "df": df,
        "within_group": rayleigh_paired_window_ttests(df, windows=windows),
        "between_group": rayleigh_between_group_ttests(df, windows=windows),
    }


def paired_ttests(df):
    results = {}
    if df.empty:
        return results

    df = df.copy()
    df["window_key"] = df["window_start_s"].astype(str) + "-" + df["window_end_s"].astype(str)
    windows = sorted(df["window_key"].unique())
    if len(windows) < 2:
        return results

    for group in sorted(df["group"].unique()):
        sub = df[df["group"] == group]
        pivot = sub.pivot_table(index="participant", columns="window_key", values="r", aggfunc="mean")
        if all(w in pivot.columns for w in windows[:2]):
            a = pivot[windows[0]].to_numpy()
            b = pivot[windows[1]].to_numpy()
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() > 1:
                test = stats.ttest_rel(a[mask], b[mask], nan_policy="omit")
                results[group] = {
                    "window_a": windows[0],
                    "window_b": windows[1],
                    "n": int(mask.sum()),
                    "t": float(test.statistic),
                    "p": float(test.pvalue),
                }
    return results


def ind_ttests(df):
    results = {}
    if df.empty:
        return results

    df = df.copy()
    df["window_key"] = df["window_start_s"].astype(str) + "-" + df["window_end_s"].astype(str)
    windows = sorted(df["window_key"].unique())

    for window in windows:
        a = df[(df["group"] == "ctrl") & (df["window_key"] == window)]["r"].to_numpy()
        b = df[(df["group"] == "pat") & (df["window_key"] == window)]["r"].to_numpy()
        if a.size > 1 and b.size > 1:
            test = stats.ttest_ind(a, b, nan_policy="omit", equal_var=False)
            results[window] = {
                "window": window,
                "n_ctrl": int(np.isfinite(a).sum()),
                "n_patient": int(np.isfinite(b).sum()),
                "t": float(test.statistic),
                "p": float(test.pvalue),
            }

    return results
