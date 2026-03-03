import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
from config import get_control_file, get_patient_file

plt.rcParams["font.family"] = "Helvetica"

def vis(participant, thr_z, merging_ms, show_plot=False, group='ctrl', plot_trials=24):
    file_path = get_control_file(participant) if group == 'ctrl' else get_patient_file(participant)
    data = np.load(file_path)  # shape: (channels, time, trials)
    fs = 200 #sampling frequency, Hz

    samples = data.shape[1]
    trials = data.shape[2]
    summaries = []

    # participant-level normalization across all cleaned trials
    speed_chunks = []
    for trial_idx in range(trials):
        trial_data = data[:, :, trial_idx]
        df_trial = pd.DataFrame(trial_data.T, columns=['x', 'y', 'pupil'])
        blink_mask = df_trial['pupil'] < -4.0
        blink_mask = ndimage.binary_dilation(blink_mask, iterations=15)
        df_trial.loc[blink_mask, ['x', 'y']] = np.nan
        speed = np.sqrt(np.diff(df_trial['x']) ** 2 + np.diff(df_trial['y']) ** 2)
        speed = speed[np.isfinite(speed)]
        if speed.size > 0:
            speed_chunks.append(speed)

    if speed_chunks:
        pooled_speed = np.concatenate(speed_chunks)
        speed_mu = np.nanmean(pooled_speed)
        speed_sigma = np.nanstd(pooled_speed)
    else:
        speed_mu, speed_sigma = np.nan, np.nan

    plot_rows = []

    for i in range(1, trials + 1):
        trial_data = data[:, :, i - 1]
        trial1 = pd.DataFrame(trial_data.T, columns=['x', 'y', 'pupil'])
        blink_mask = trial1['pupil'] < -4.0
        blink_mask = ndimage.binary_dilation(blink_mask, iterations=15)
        trial1.loc[blink_mask, ['x', 'y', 'pupil']] = np.nan
        x = trial1['x']
        y = trial1['y']

        xdiff = np.diff(x) ** 2
        ydiff = np.diff(y) ** 2
        speed = np.sqrt(xdiff + ydiff)
        if np.isfinite(speed_sigma) and speed_sigma > 0 and np.isfinite(speed_mu):
            speed_z = (speed - speed_mu) / speed_sigma
        else:
            speed_z = np.full_like(speed, np.nan, dtype=float)
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
            'trial': i,
            'n_saccades': int(n),
            'avg_duration_ms': avg_duration,
            'speed_var': np.nanvar(speed)
        })

        # collect plotting data; render once at the end
        if show_plot:
            peak = np.nanmax(speed_z) if np.isfinite(speed_z).any() else 1.0
            plot_rows.append((i, time, speed_z, saccades * peak))

    if show_plot and plot_rows:
        n_cols = 4
        page_size = max(1, int(plot_trials))
        n_pages = int(np.ceil(len(plot_rows) / page_size))

        for page_idx in range(n_pages):
            start = page_idx * page_size
            end = min((page_idx + 1) * page_size, len(plot_rows))
            page_rows = plot_rows[start:end]
            n_plot = len(page_rows)
            n_rows = int(np.ceil(n_plot / n_cols))

            # Compact sizing to keep each 24-trial page on-screen.
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(14, min(9, 2.1 * n_rows)),
                sharex=True,
                sharey=False
            )
            axes = np.atleast_1d(axes).ravel()

            for ax_idx in range(n_plot):
                trial_num, time, speed_z, movement_trace = page_rows[ax_idx]
                ax = axes[ax_idx]
                ax.plot(time, speed_z, lw=0.8)
                ax.plot(time, movement_trace, lw=0.8)
                yvals = np.concatenate([
                    speed_z[np.isfinite(speed_z)],
                    movement_trace[np.isfinite(movement_trace)]
                ])
                if yvals.size > 0:
                    ymin = np.min(yvals)
                    ymax = np.max(yvals)
                    if ymax > ymin:
                        pad = 0.05 * (ymax - ymin)
                        ax.set_ylim(ymin - pad, ymax + pad)
                ax.set_title(f'Trial {trial_num}', fontsize=9)
                ax.set_ylabel('z')

            for ax in axes[n_plot:]:
                ax.set_visible(False)

            fig.suptitle(
                f'Participant {participant} ({group}) - trials {page_rows[0][0]} to {page_rows[-1][0]}',
                fontsize=12
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    # after iterating all trials: concise output only
    summary_df = pd.DataFrame(summaries).sort_values('trial').reset_index(drop=True)
    print('\nPer-trial summary:')
    for r in summary_df.itertuples(index=False):
        print(f"trial {r.trial}: saccades = {r.n_saccades}, avg duration = {r.avg_duration_ms:.1f} ms, speed var = {r.speed_var}")

    return summary_df
