import numpy as np
from mean_results import con_means, pat_means

def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.std(x, ddof=1) / np.sqrt(len(x))

c = con_means(show_plots=False)
p = pat_means(show_plots=False)

sacc_rate = np.concatenate([c["rates"], p["rates"]])                       # Hz
sacc_dur_ms = np.concatenate([c["durations"], p["durations"]])   # ms
fix_rate  = np.concatenate([c["fixation_rates"], p["fixation_rates"]])     # Hz
fix_dur_ms = np.concatenate([c["fixation_durations"], p["fixation_durations"]])  # ms

print("Saccade rate (Hz):", np.nanmean(sacc_rate), "±", sem(sacc_rate))
print("Saccade duration (ms):", np.nanmean(sacc_dur_ms), "±", sem(sacc_dur_ms))
print("Fixation rate (Hz):", np.nanmean(fix_rate), "±", sem(fix_rate))
print("Fixation duration (ms):", np.nanmean(fix_dur_ms), "±", sem(fix_dur_ms))

# saccade speed metrics
mean_speed = np.concatenate([c["mean_speeds"], p["mean_speeds"]])
max_speed  = np.concatenate([c["max_speeds"], p["max_speeds"]])

print("Mean saccade speed:", np.nanmean(mean_speed), "±", sem(mean_speed))
print("Peak saccade speed:", np.nanmean(max_speed), "±", sem(max_speed))