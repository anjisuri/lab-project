import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure

from config import get_control_file


plt.rcParams["font.family"] = "Helvetica"

FS = 200  # sampling frequency, Hz


def _participant_speed_stats(data):
    speed_chunks = []
    n_trials = data.shape[2]
    for trial_idx in range(n_trials):
        trial_data = data[:, :, trial_idx]
        df = pd.DataFrame(trial_data.T, columns=["x", "y", "pupil"])
        blink_mask = df["pupil"] < -4.0
        blink_mask = ndimage.binary_dilation(blink_mask, iterations=15)
        df.loc[blink_mask, ["x", "y"]] = np.nan
        speed = np.sqrt(np.diff(df["x"]) ** 2 + np.diff(df["y"]) ** 2)
        speed = speed[np.isfinite(speed)]
        if speed.size > 0:
            speed_chunks.append(speed)
    if not speed_chunks:
        return np.nan, np.nan
    pooled = np.concatenate(speed_chunks)
    return np.nanmean(pooled), np.nanstd(pooled)


def vis_control_trial(participant_id=14, trial_number=39, thr_z=1.5, merge_ms=10):
    file_path = get_control_file(participant_id)
    data = np.load(file_path)  # shape: (channels, time, trials)
    n_trials = data.shape[2]

    if trial_number < 1 or trial_number > n_trials:
        raise ValueError(f"trial_number must be in [1, {n_trials}]")

    trial_data = data[:, :, trial_number - 1]
    df = pd.DataFrame(trial_data.T, columns=["x", "y", "pupil"])

    blink_mask = df["pupil"] < -4.0
    blink_mask = ndimage.binary_dilation(blink_mask, iterations=15)
    df_clean = df.copy()
    df_clean.loc[blink_mask, ["x", "y", "pupil"]] = np.nan

    mu, sigma = _participant_speed_stats(data)

    x = df_clean["x"]
    y = df_clean["y"]
    speed = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
        speed_z = (speed - mu) / sigma
    else:
        speed_z = np.full_like(speed, np.nan, dtype=float)

    time = np.arange(len(speed)) / FS

    plt.figure(figsize=(10, 4))
    plt.plot(time, speed_z)
    plt.title(f"Instantaneous eye movement speed (ctrl {participant_id}, trial {trial_number})")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed, zscored (a.u.)")
    plt.tight_layout()

    iters = max(1, int((merge_ms / 1000.0) * FS))
    raw = speed_z > thr_z
    print("raw > thr_z:", int(raw.sum()))
    print("iters (samples merged):", iters)

    saccades = ndimage.binary_dilation(raw, iterations=iters)
    labelled, n = ndimage.label(saccades.astype(np.uint8))
    regions = ndimage.find_objects(labelled)

    for region in regions:
        start = time[region[0].start].round(4)
        end = time[region[0].stop - 1].round(4)
        duration_ms = ((end - start) * 1000).round(4)
        print(f"{start}s -> {end}s  (duration {duration_ms}ms)")

    print(f"number of saccades = {n}")
    peak = np.nanmax(speed_z) if np.isfinite(speed_z).any() else 1.0
    plt.plot(time, saccades * peak)
    plt.show()

    labels = measure.label(saccades.astype(np.uint8), connectivity=1)
    props = measure.regionprops_table(
        labels[:, None],
        intensity_image=speed_z[:, None],
        properties=("label", "area", "bbox", "mean_intensity", "max_intensity"),
    )

    stats_df = pd.DataFrame(props).rename(
        columns={
            "area": "n_samples",
            "bbox-0": "start_idx",
            "bbox-2": "end_idx_exclusive",
            "mean_intensity": "mean_speed",
            "max_intensity": "max_speed",
        }
    )

    if stats_df.empty:
        print("number of saccades = 0")
        return stats_df

    stats_df["end_idx"] = stats_df["end_idx_exclusive"] - 1
    stats_df["start_s"] = stats_df["start_idx"] / FS
    stats_df["end_s"] = stats_df["end_idx"] / FS
    stats_df["duration_ms"] = (stats_df["n_samples"] / FS) * 1000

    print(f"number of saccades = {len(stats_df)}")
    print(stats_df[["label", "start_s", "end_s", "duration_ms", "mean_speed", "max_speed"]])
    return stats_df


if __name__ == "__main__":
    vis_control_trial(participant_id=10, trial_number=1)
