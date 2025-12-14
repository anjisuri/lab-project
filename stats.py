from scipy import stats
from mean_results import con_means, pat_means

controls = con_means(show_plots=False)
patients = pat_means(show_plots=False)

metrics = ["rates", "durations", "mean_speeds", "max_speeds"]

for metric in metrics:
    control_vals = controls[metric]
    patient_vals = patients[metric]
    length = min(len(control_vals), len(patient_vals))
    if length == 0:
        print(f"{metric}: not enough data for t-test")
        continue
    result = stats.ttest_rel(control_vals[:length], patient_vals[:length], nan_policy="omit")
    print(f"\n{metric}: {result}")