from scipy import stats
from mean_results import con_means, pat_means, con_means_window, pat_means_window
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import numpy as np
from pathlib import Path

# EYE MOVEMENT STATS

controls = con_means(show_plots=False)
patients = pat_means(show_plots=False)

metrics = ["rates", "durations", "mean_speeds", "max_speeds", "fixation_rates", "fixation_durations"]

for metric in metrics:
    control_vals = controls[metric]
    patient_vals = patients[metric]
    if len(control_vals) == 0 or len(patient_vals) == 0:
        print(f"{metric}: not enough data for t-test")
        continue
    # Between-group comparison (independent samples)
    result = stats.ttest_ind(control_vals, patient_vals, nan_policy="omit", equal_var=False)
    print(f"\n{metric}: {result}")


# MEG STATS

def anova(DV):

    con_pre = con_means_window(1, 4, show_plots=False, return_average=False)
    pat_pre = pat_means_window(1, 4, show_plots=False, return_average=False)
    con_stim = con_means_window(4, 7, show_plots=False, return_average=False)
    pat_stim = pat_means_window(4, 7, show_plots=False, return_average=False)

    rows = []
    all_data = [
        ("control", "prestim", con_pre),
        ("control", "stim", con_stim),
        ("patient", "prestim", pat_pre),
        ("patient", "stim", pat_stim),
    ]
    for group, condition, data in all_data:
        for participant_id, value in enumerate(data[DV], start=1):
            rows.append(
                {
                    "participant_id": participant_id,
                    "group": group,
                    "condition": condition,
                    DV: value,
                }
            )

    df = pd.DataFrame(rows)

    counts = df.groupby(["participant_id", "group"])["condition"].nunique()
    keep = counts[counts == 2].index
    df = df.set_index(["participant_id", "group"]).loc[keep].reset_index()

    model = ols(f"{DV} ~ C(group) * C(condition)", data=df).fit()
    anova_result = sm.stats.anova_lm(model, type=2)
    
    print(df, '\n', anova_result)


#paired t-test, uses fixation duration vals from ANOVA input table 
def paired_fixation_duration_posthoc(final=True):
    table_path = Path("analysis_outputs/anova_inputs/fixation_duration_ms.csv")
    df = pd.read_csv(table_path)

    fix_vals = df["fixation"].to_numpy(dtype=float)
    cue_vals = df["cue"].to_numpy(dtype=float)
    mask = np.isfinite(fix_vals) & np.isfinite(cue_vals)
    fix_vals = fix_vals[mask]
    cue_vals = cue_vals[mask]

    if len(fix_vals) < 2:
        out = {"n": int(len(fix_vals)), "t": np.nan, "df": np.nan, "p": np.nan}
        if final:
            print(out)
        return out

    test = stats.ttest_rel(fix_vals, cue_vals, nan_policy="omit")
    out = {
        "n": int(len(fix_vals)),
        "t": float(test.statistic),
        "df": int(len(fix_vals) - 1),
        "p": float(test.pvalue),
        "mean_fixation": float(np.mean(fix_vals)),
        "mean_cue": float(np.mean(cue_vals)),
        "mean_diff_fix_minus_cue": float(np.mean(fix_vals - cue_vals)),
    }
    return out