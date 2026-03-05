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

def _print_metric(label, csv_name, unit):
    df = pd.read_csv(ANOVA_DIR / csv_name)
    fix_vals = df["fixation"].to_numpy(dtype=float)
    cue_vals = df["cue"].to_numpy(dtype=float)
    pooled = np.concatenate([fix_vals, cue_vals])
    print(f"{label} ({unit})")
    print(f"  Fixation: {np.nanmean(fix_vals)} ± {sem(fix_vals)} (n={len(df)})")
    print(f"  Cue: {np.nanmean(cue_vals)} ± {sem(cue_vals)} (n={len(df)})")
    print(f"  Pooled (fix+cue): {np.nanmean(pooled)} ± {sem(pooled)} (n={len(pooled)})")

print("Windowed means ± SEM (ANOVA-matched paired cohort):")
_print_metric("Saccade rate", "saccade_frequency.csv", "Hz")
_print_metric("Saccade duration", "saccade_duration_ms.csv", "ms")
_print_metric("Fixation rate", "fixation_frequency.csv", "Hz")
_print_metric("Fixation duration", "fixation_duration_ms.csv", "ms")
_print_metric("Mean saccade speed", "mean_saccade_speed.csv", "z")
_print_metric("Peak saccade speed", "max_saccade_speed.csv", "z")
