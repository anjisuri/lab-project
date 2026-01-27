from scipy import stats
from mean_results import con_means, pat_means, con_means_window, pat_means_window
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd

# SACCADE STATS

controls = con_means(show_plots=False)
patients = pat_means(show_plots=False)

metrics = ["rates", "durations", "mean_speeds", "max_speeds", "fixation_rates", "fixation_durations"]

for metric in metrics:
    control_vals = controls[metric]
    patient_vals = patients[metric]
    length = min(len(control_vals), len(patient_vals))
    if length == 0:
        print(f"{metric}: not enough data for t-test")
        continue
    result = stats.ttest_rel(control_vals[:length], patient_vals[:length], nan_policy="omit")
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
