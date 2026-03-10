from pathlib import Path
import numpy as np
import pandas as pd

ANOVA_DIR = Path("analysis_outputs/anova_inputs")

def sem(x):
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan
    return np.std(arr, ddof=1) / np.sqrt(len(arr))

# metric_name options:
# - "saccade_frequency"
# - "saccade_duration_ms"
# - "fixation_frequency"
# - "fixation_duration_ms"
# - "mean_saccade_speed"
# - "max_saccade_speed"
# - "phase_locking_saccade_onset_r"
# - "phase_locking_fixation_onset_r"

def metric(metric_name):
    df = pd.read_csv(ANOVA_DIR / f"{str(metric_name)}.csv")
    out = {}
    for group in ("control", "patient"):
        sub = df[df["group"] == group]
        fix_vals = sub["fixation"].to_numpy(dtype=float)
        cue_vals = sub["cue"].to_numpy(dtype=float)
        out[group] = {
            "fixation": f"{np.nanmean(fix_vals):.3f} ± {sem(fix_vals):.3f}",
            "cue": f"{np.nanmean(cue_vals):.3f} ± {sem(cue_vals):.3f}",
        }
    return out
