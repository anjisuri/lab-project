# save as make_anova_tables.py
import os
import numpy as np
import pandas as pd
from mean_results import con_means_window, pat_means_window
from rayleigh import rayleigh_window_group_df
from config import get_common_participant_ids_by_group

OUT_DIR = "analysis_outputs/anova_inputs"
os.makedirs(OUT_DIR, exist_ok=True)

# windows: fixation = 1-4s, cue = 4-7s
con_fix = con_means_window(1, 4, show_plots=False, return_average=False)
con_cue = con_means_window(4, 7, show_plots=False, return_average=False)
pat_fix = pat_means_window(1, 4, show_plots=False, return_average=False)
pat_cue = pat_means_window(4, 7, show_plots=False, return_average=False)


COMMON_IDS = get_common_participant_ids_by_group(windows=((1, 4), (4, 7)))
SACC_CTRL_COMMON = COMMON_IDS["ctrl"]
SACC_PAT_COMMON = COMMON_IDS["patient"]

def export_metric(key, out_name):
    rows = []

    ctrl_fix = dict(zip(con_fix["valid_ids"], con_fix[key]))
    ctrl_cue = dict(zip(con_cue["valid_ids"], con_cue[key]))
    ctrl_common = SACC_CTRL_COMMON
    for pid in ctrl_common:
        a = ctrl_fix[pid]
        b = ctrl_cue[pid]
        if np.isfinite(a) and np.isfinite(b):
            rows.append(
                {
                    "subject": f"C{int(pid):02d}",
                    "group": "control",
                    "fixation": float(a),
                    "cue": float(b),
                }
            )

    pat_fix_map = dict(zip(pat_fix["valid_ids"], pat_fix[key]))
    pat_cue_map = dict(zip(pat_cue["valid_ids"], pat_cue[key]))
    pat_common = SACC_PAT_COMMON
    for pid in pat_common:
        a = pat_fix_map[pid]
        b = pat_cue_map[pid]
        if np.isfinite(a) and np.isfinite(b):
            rows.append(
                {
                    "subject": f"P{int(pid):02d}",
                    "group": "patient",
                    "fixation": float(a),
                    "cue": float(b),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["group", "subject"]).reset_index(drop=True)
    df.to_csv(f"{OUT_DIR}/{out_name}.csv", index=False)
    print(f"Wrote {OUT_DIR}/{out_name}.csv  (n={len(df)})")


def export_phase_locking(which, out_name):
    windows = ((1, 4), (4, 7))
    df_ctrl = rayleigh_window_group_df(group="ctrl", which=which, windows=windows, final=False)
    df_pat = rayleigh_window_group_df(group="patient", which=which, windows=windows, final=False)
    df = pd.concat([df_ctrl, df_pat], ignore_index=True)

    # ANOVA-ready wide table on subject-level resultant vector length (r)
    rows = []
    for group_code, group_label, prefix, allowed_ids in [
        ("ctrl", "control", "C", set(SACC_CTRL_COMMON)),
        ("patient", "patient", "P", set(SACC_PAT_COMMON)),
    ]:
        sub = df[df["group"] == group_code]
        pivot_r = sub.pivot_table(
            index="participant",
            columns=["window_start_s", "window_end_s"],
            values="r",
            aggfunc="mean",
        )
        for participant in sorted(pivot_r.index):
            if participant not in allowed_ids:
                continue
            fix_key = (1, 4)
            cue_key = (4, 7)
            if fix_key not in pivot_r.columns or cue_key not in pivot_r.columns:
                continue
            fix_r = pivot_r.loc[participant, fix_key]
            cue_r = pivot_r.loc[participant, cue_key]
            if np.isfinite(fix_r) and np.isfinite(cue_r):
                rows.append(
                    {
                        "subject": f"{prefix}{int(participant):02d}",
                        "group": group_label,
                        "fixation": float(fix_r),
                        "cue": float(cue_r),
                    }
                )

    wide_df = pd.DataFrame(rows).sort_values(["group", "subject"]).reset_index(drop=True)
    wide_path = f"{OUT_DIR}/{out_name}.csv"
    wide_df.to_csv(wide_path, index=False)
    print(f"Wrote {wide_path}  (n={len(wide_df)})")

# required 4 metrics
export_metric("rates", "saccade_frequency")
export_metric("durations", "saccade_duration_ms")
export_metric("mean_speeds", "mean_saccade_speed")
export_metric("max_speeds", "max_saccade_speed")
export_metric("fixation_rates", "fixation_frequency")
export_metric("fixation_durations", "fixation_duration_ms")

# phase-locking (resultant vector length r) for both onset types
export_phase_locking(which="start", out_name="phase_locking_saccade_onset_r")
export_phase_locking(which="end", out_name="phase_locking_fixation_onset_r")
