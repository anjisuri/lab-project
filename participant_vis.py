import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import zscore
from skimage import measure

def vis(participant, thr_z, merging_ms, show_plot=False):
    df = pd.read_csv(f'/Users/anji/Desktop/lab project/python_data/controls/participant_{participant}.csv')
    fs = 200 #sampling frequency, Hz

    samples = 1601
    trials = df.shape[0] // samples
    df['trial'] = np.repeat(np.arange(1, trials + 1), samples)
    summaries = []

    for i in df['trial'].unique():
        trial1 = df[df['trial'] == i]
        x = trial1['x']
        y = trial1['y']
        pupil = trial1['pupil']

        xdiff = np.diff(x) ** 2
        ydiff = np.diff(y) ** 2
        speed = np.sqrt(xdiff + ydiff)
        speed_z = zscore(speed)
        time = np.arange(len(speed)) / fs  # seconds (fs = 200Hz)

        # --- saccade statistics ---
        iters = max(1, int((merging_ms / 1000.0) * fs))   # samples to merge at fs Hz

        raw = speed_z > thr_z
        # merge short gaps between supra-threshold samples
        saccades = ndimage.binary_dilation(raw, iterations=iters)
        # optional gentle cleanup to remove isolated single-sample noise (commented by default)
        # saccades = ndimage.binary_opening(saccades)

        labelled, n = ndimage.label(saccades.astype(np.uint8))

        # Build skimage props per trial
        labels_arr = measure.label(saccades.astype(np.uint8), connectivity=1)
        props = measure.regionprops_table(
            labels_arr[:, None],
            intensity_image=speed_z[:, None],
            properties=('label', 'area', 'bbox', 'mean_intensity', 'max_intensity')
        )
        stats_df = pd.DataFrame(props).rename(
            columns={
                'area': 'n_samples',
                'bbox-0': 'start_idx',
                'bbox-2': 'end_idx_exclusive',
                'mean_intensity': 'mean_speed_z',
                'max_intensity': 'max_speed_z'
            }
        )
        if not stats_df.empty:
            stats_df['end_idx'] = stats_df['end_idx_exclusive'] - 1
            stats_df['start_s'] = stats_df['start_idx'] / fs
            stats_df['end_s'] = stats_df['end_idx'] / fs
            stats_df['duration_ms'] = (stats_df['n_samples'] / fs) * 1000
            avg_duration = float(stats_df['duration_ms'].mean())
        else:
            avg_duration = 0.0

        # Save concise per-trial summary only
        summaries.append({
            'trial': int(i),
            'n_saccades': int(n),
            'avg_duration_ms': avg_duration,
            'speed_var': np.var(speed)
        })

        # plot per trial (off by default)
        if show_plot:
            movement = saccades
            plt.figure(figsize=(10, 4))
            plt.plot(time, speed_z)
            plt.title(f'Instantaneous eye movement speed - trial {i}')
            plt.xlabel('Time (s)')
            plt.ylabel('Speed, z-scored (a.u.)')
            plt.tight_layout()
            plt.plot(time, movement * np.nanmax(speed_z))
            plt.show()

    # after iterating all trials: concise output only
    summary_df = pd.DataFrame(summaries).sort_values('trial').reset_index(drop=True)
    print('\nPer-trial summary:')
    for r in summary_df.itertuples(index=False):
        print(f"trial {r.trial}: saccades = {r.n_saccades}, avg duration = {r.avg_duration_ms:.1f} ms, speed var = {r.speed_var}")

    return summary_df