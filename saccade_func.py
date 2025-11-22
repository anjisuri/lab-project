import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from skimage import measure
import pandas as pd

def trial(participant, trial_number, show_plot=True, show_stats=True):
    # loading
    file_path = f'/Users/anji/Desktop/lab project/EyeData/controls/ctrl_{participant}.npy'
    data = np.load(file_path)  # shape: (channels, time, trials)
    trial_idx = trial_number - 1
    trial_data = data[:, :, trial_idx]
    df = pd.DataFrame(trial_data.T, columns=['x', 'y', 'pupil']) # transpose from (channels, time)
    
    fs = 200
    samples = data.shape[1]
    trials = data.shape[2]

    # blink exclusion
    blink_mask = df['pupil'] < -4.0 # True = blink, False = valid
    blink_mask = ndimage.binary_dilation(blink_mask, iterations=15)
    time = np.arange(samples) / fs
    df_clean = df.copy()
    df_clean.loc[blink_mask, :] = np.nan

    merging_ms = 10
    thr_z = 2

    x = df_clean['x']
    y = df_clean['y']
    pupil = df_clean['pupil']

    xdiff = np.diff(x) ** 2
    ydiff = np.diff(y) ** 2
    speed = np.sqrt(xdiff + ydiff)
    speed_z = stats.zscore(speed, nan_policy = 'omit')
    time = np.arange(len(speed)) / fs  # seconds (fs = 200Hz)

    if show_plot:
        plt.figure(figsize=(10,4))

        plt.plot(time, speed_z)


        plt.title('Instantaneous eye movement speed')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed, zscored (a.u.)')
        plt.tight_layout()

    # --- saccade statistics ---
    iters = max(1, int((merging_ms / 1000.0) * fs))   # samples to merge at fs Hz

    raw = speed_z > thr_z
    
    # merge short gaps between supra-threshold samples
    saccades = ndimage.binary_dilation(raw, iterations=iters)

    labelled, n = ndimage.label(saccades.astype(np.uint8))

    movement = saccades
    print(f'number of saccades = {n}')

    if show_plot:
        plt.plot(time, movement*np.nanmax(speed_z))

        plt.show()

    labels = measure.label(movement.astype(np.uint8), connectivity=1)
    props = measure.regionprops_table(
        labels[:, None],                 # make it 2-D for skimage
        intensity_image=speed_z[:, None],  # to get mean/max speed per saccade
        properties=('label', 'area', 'bbox', 'mean_intensity', 'max_intensity')
    )

    stats_df = pd.DataFrame(props).rename(
        columns = {
            'area': 'n_samples',
            'bbox-0': 'start_idx',
            'bbox-2': 'end_idx_exclusive',
            'mean_intensity': 'mean_speed',
            'max_intensity': 'max_speed'
        }
    )

    if show_stats:
        stats_df['end_idx'] = stats_df['end_idx_exclusive'] - 1
        stats_df['start_s'] = stats_df['start_idx'] / fs
        stats_df['end_s'] = stats_df['end_idx'] / fs
        stats_df['duration_ms'] = (stats_df['n_samples'] / fs) * 1000
        print(stats_df[['label','start_s','end_s','duration_ms','mean_speed','max_speed']])

def participant(participant):
    file_path = f'/Users/anji/Desktop/lab project/EyeData/controls/ctrl_{participant}.npy'
    data = np.load(file_path)  # shape (channels, time, trials)
    n_trials = data.shape[2]

    for trial_num in range(1, n_trials+1):
        print(f"\n=== Trial {trial_num} ===")
        trial(participant, trial_num, show_plot=False, show_stats=False)