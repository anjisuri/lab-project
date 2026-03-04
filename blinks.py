import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from config import get_control_file

plt.rcParams["font.family"] = "Helvetica"

def _participant_pupil_stats(data):
    pupil_chunks = []
    n_trials = data.shape[2]
    for trial_idx in range(n_trials):
        trial_pupil = data[2, :, trial_idx].astype(float)
        p_mu = np.nanmean(trial_pupil)
        p_sigma = np.nanstd(trial_pupil)
        if np.isfinite(p_mu) and np.isfinite(p_sigma) and p_sigma > 0:
            trial_pupil_z = (trial_pupil - p_mu) / p_sigma
        else:
            trial_pupil_z = np.full_like(trial_pupil, np.nan, dtype=float)
        blink_mask = trial_pupil_z < -2.0
        blink_mask = ndimage.binary_dilation(blink_mask, iterations=15)
        clean = trial_pupil.copy()
        clean[blink_mask] = np.nan
        clean = clean[np.isfinite(clean)]
        if clean.size > 0:
            pupil_chunks.append(clean)

    if pupil_chunks:
        pooled = np.concatenate(pupil_chunks)
        return float(np.nanmean(pooled)), float(np.nanstd(pooled))
    return np.nan, np.nan

# plotting pupil dilation for a single trial (to identify eye blinks)
def pupil_dilation(participant, trial_number):
    fs = 200  # sampling frequency Hz

    # load npy file for participant
    file_path = get_control_file(participant)
    data = np.load(file_path)  # shape: (3, time, trials)

    pupil_mu, pupil_sigma = _participant_pupil_stats(data)
    trial_pupil = data[2, :, trial_number - 1].astype(float)  # pupil channel for this trial
    if np.isfinite(pupil_mu) and np.isfinite(pupil_sigma) and pupil_sigma > 0:
        trial_pupil_z = (trial_pupil - pupil_mu) / pupil_sigma
    else:
        trial_pupil_z = np.full_like(trial_pupil, np.nan, dtype=float)
    blink_mask = trial_pupil_z < -2.0
    blink_mask = ndimage.binary_dilation(blink_mask, iterations=15)

    samples = trial_pupil.shape[0]
    time = np.arange(samples) / fs

    plt.figure(figsize=(10,4))
    plt.plot(time, trial_pupil_z, lw=1.0, label="pupil z")
    plt.plot(time, np.where(blink_mask, trial_pupil_z, np.nan), color="red", lw=1.0, label="blink")
    plt.title(f'Pupil dilation (z): participant {participant} trial {trial_number}')
    plt.xlabel('Time (s)')
    plt.ylabel('Pupil (z)')
    plt.legend()
    plt.tight_layout()
    plt.show()
