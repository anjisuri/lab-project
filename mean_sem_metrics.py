import numpy as np
from mean_results import con_means, pat_means

def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.std(x, ddof=1) / np.sqrt(len(x))

c = con_means(show_plots=False)
p = pat_means(show_plots=False)

sacc_rate = np.concatenate([c["rates"], p["rates"]])                       # Hz
sacc_dur_s = np.concatenate([c["durations"], p["durations"]]) / 1000.0     # s
fix_rate  = np.concatenate([c["fixation_rates"], p["fixation_rates"]])     # Hz
fix_dur_s = np.concatenate([c["fixation_durations"], p["fixation_durations"]]) / 1000.0  # s

print("Saccade rate (Hz):", np.nanmean(sacc_rate), "±", sem(sacc_rate))
print("Saccade duration (s):", np.nanmean(sacc_dur_s), "±", sem(sacc_dur_s))
print("Fixation rate (Hz):", np.nanmean(fix_rate), "±", sem(fix_rate))
print("Fixation duration (s):", np.nanmean(fix_dur_s), "±", sem(fix_dur_s))

# saccade speed metrics
mean_speed = np.concatenate([c["mean_speeds"], p["mean_speeds"]])
max_speed  = np.concatenate([c["max_speeds"], p["max_speeds"]])

print("Mean saccade speed:", np.nanmean(mean_speed), "±", sem(mean_speed))
print("Peak saccade speed:", np.nanmean(max_speed), "±", sem(max_speed))