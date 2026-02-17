# save as make_anova_tables.py
import os
import numpy as np
import pandas as pd
from mean_results import con_means_window, pat_means_window

OUT_DIR = "analysis_outputs/anova_inputs"
os.makedirs(OUT_DIR, exist_ok=True)

# windows: fixation = 1-4s, cue = 4-7s
con_fix = con_means_window(1, 4, show_plots=False, return_average=False)
con_cue = con_means_window(4, 7, show_plots=False, return_average=False)
pat_fix = pat_means_window(1, 4, show_plots=False, return_average=False)
pat_cue = pat_means_window(4, 7, show_plots=False, return_average=False)

def export_metric(key, out_name):
    rows = []

    for i, (a, b) in enumerate(zip(con_fix[key], con_cue[key]), start=1):
        if np.isfinite(a) and np.isfinite(b):
            rows.append({"subject": f"C{i:02d}", "group": "control", "fixation": float(a), "cue": float(b)})

    for i, (a, b) in enumerate(zip(pat_fix[key], pat_cue[key]), start=1):
        if np.isfinite(a) and np.isfinite(b):
            rows.append({"subject": f"P{i:02d}", "group": "patient", "fixation": float(a), "cue": float(b)})

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/{out_name}.csv", index=False)
    print(f"Wrote {OUT_DIR}/{out_name}.csv  (n={len(df)})")

# required 4 metrics
export_metric("rates", "saccade_frequency")
export_metric("durations", "saccade_duration_ms")
export_metric("fixation_rates", "fixation_frequency")
export_metric("fixation_durations", "fixation_duration_ms")
