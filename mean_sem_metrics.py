from pathlib import Path
import numpy as np
import pandas as pd

ANOVA_DIR = Path("analysis_outputs/anova_inputs")
ALL_METRICS = (
    "saccade_frequency",
    "saccade_duration_ms",
    "fixation_frequency",
    "fixation_duration_ms",
    "mean_saccade_speed",
    "max_saccade_speed",
    "phase_locking_saccade_onset_r",
    "phase_locking_fixation_onset_r",
)

def sem(x):
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan
    return np.std(arr, ddof=1) / np.sqrt(len(arr))

def metric(metric_name):
    df = pd.read_csv(ANOVA_DIR / f"{str(metric_name)}.csv")
    out = {}
    for group in ("control", "patient"):
        sub = df[df["group"] == group]
        fix_vals = sub["fixation"].to_numpy(dtype=float)
        cue_vals = sub["cue"].to_numpy(dtype=float)
        out[group] = {
            "fixation": f"{np.nanmean(fix_vals):.4f} ± {sem(fix_vals):.4f}",
            "cue": f"{np.nanmean(cue_vals):.4f} ± {sem(cue_vals):.4f}",
        }
    return out


def run_all(metrics=ALL_METRICS):
    results = {}
    for metric_name in metrics:
        results[metric_name] = metric(metric_name)
    return results


if __name__ == "__main__":
    all_results = run_all()
    for metric_name, vals in all_results.items():
        print(metric_name)
        for group in ("control", "patient"):
            group_vals = vals[group]
            print(f"  {group}: fixation={group_vals['fixation']} | cue={group_vals['cue']}")
        print()
