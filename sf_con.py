import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
import pandas as pd
from config import get_control_file

def _participant_speed_stats(data):
    n_trials = data.shape[2]
    speed_chunks = []
    for trial_idx in range(n_trials):
        trial_data = data[:, :, trial_idx]
        df = pd.DataFrame(trial_data.T, columns=['x', 'y', 'pupil'])
        blink_mask = df['pupil'] < -4.0
        blink_mask = ndimage.binary_dilation(blink_mask, iterations=15)
        df.loc[blink_mask, ['x', 'y']] = np.nan
        speed = np.sqrt(np.diff(df['x']) ** 2 + np.diff(df['y']) ** 2)
        speed = speed[np.isfinite(speed)]
        if speed.size > 0:
            speed_chunks.append(speed)

    if speed_chunks:
        pooled_speed = np.concatenate(speed_chunks)
        return np.nanmean(pooled_speed), np.nanstd(pooled_speed)
    return np.nan, np.nan

def trial_vis(participant, trial_number):
    file_path = get_control_file(participant)
    data = np.load(file_path)  # shape: (channels, time, trials)
    trial_idx = trial_number - 1
    trial_data = data[:, :, trial_idx]
    fs = 200  # sampling frequency Hz
    samples = 1601

    df = pd.DataFrame(trial_data.T, columns=['x', 'y', 'pupil']) # transpose from (channels, time)
    time = np.arange(samples) / fs

    plt.figure(figsize = (10,4))
    plt.subplot(2,1,1)
    plt.title('Eye movement')
    plt.plot(time, df['x'], label = 'x position')
    plt.subplot(2,1,1)
    plt.plot(time, df['y'], label = 'y position')
    plt.xlabel('Time (s)')
    plt.legend(loc = 'upper right', ncol = 2)

    plt.subplot(2,1,2)
    plt.title('Pupil dilation')
    plt.plot(time, df['pupil'])
    plt.xlabel('Time (s)')
    plt.tight_layout()

    plt.show()

def trial(participant, trial_number, show_plot=True, show_stats=True, final=True, speed_mu=None, speed_sigma=None):
    # loading
    file_path = get_control_file(participant)
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
    if speed_mu is None or speed_sigma is None:
        speed_mu, speed_sigma = _participant_speed_stats(data)

    if np.isfinite(speed_sigma) and speed_sigma > 0 and np.isfinite(speed_mu):
        speed_z = (speed - speed_mu) / speed_sigma
    else:
        speed_z = np.full_like(speed, np.nan, dtype=float)
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
    movement = ndimage.binary_dilation(raw, iterations=iters)

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

    stats_df = stats_df[stats_df['n_samples'] >= 0.01 * fs].copy()  # >=10 ms at fs=200

    min_samples = int(np.ceil(0.01*fs))
    max_samples = int(np.floor(0.090 * fs))  # 90 ms gate

    stats_df['too_short'] = stats_df['n_samples'] < min_samples
    stats_df['too_long'] = stats_df['n_samples'] > max_samples

    # keep all rows for printing, but only longer ones in the mask/plot
    keep_labels = stats_df.loc[~(stats_df['too_short'] | stats_df['too_long']), 'label'].to_numpy()
    movement_filtered = np.isin(labels, keep_labels)

    kept_mask = ~(stats_df['too_short'] | stats_df['too_long'])
    final_count = kept_mask.sum()
    removed_counts = {
        'too_short': int(stats_df['too_short'].sum()),
        'too_long': int(stats_df['too_long'].sum()),
    }
    if final:
        print(f'final saccade count = {final_count}')
        print('removed:', removed_counts)

    stats_df['end_idx'] = stats_df['end_idx_exclusive'] - 1
    stats_df['start_s'] = stats_df['start_idx'] / fs
    stats_df['end_s'] = stats_df['end_idx'] / fs
    stats_df['duration_ms'] = (stats_df['n_samples'] / fs) * 1000

    # guard for lack of valid saccades
    has_data = np.isfinite(speed_z).any() and movement_filtered.any()
    if show_plot and has_data:
        peak = np.nanmax(speed_z)
        plt.plot(time, movement * peak, 'g--', linewidth=0.75)
        plt.plot(time, movement_filtered * peak)
        plt.show()

    if show_stats and not stats_df.empty:
        print(stats_df[['label','start_s','end_s','duration_ms','mean_speed','max_speed','too_short','too_long']])
    
    # create valid df for averaging in participant function
    valid_stats = stats_df.loc[
        ~(stats_df['too_short'] | stats_df['too_long']),
        ['start_idx', 'end_idx', 'start_s', 'end_s', 'duration_ms', 'mean_speed', 'max_speed'],
    ]

    trial_rate = len(valid_stats) / (samples / fs)

    no_saccades = len(valid_stats) == 0

    if not no_saccades:
        stdx = np.nanstd(x)
        stdy = np.nanstd(y)
        stdpup = np.nanstd(pupil)
        pupil_jitter = np.nanstd(np.diff(pupil))
        # speed_peak = np.nanpercentile(np.abs(speed_z), 99)

        qc = {
            'too_many_blinks': np.mean(blink_mask) > 0.5,
            'pupil_noise': pupil_jitter > 1.0,
            'flat_pupil': stdpup == 0,
            # 'speed_noise': speed_peak > 6,
            'flat_x': stdx < 0.1,
            'flat_y': stdy < 0.1,
            'x_noise': stdx > 4,
            'y_noise': stdy > 4,
            'no_saccades': False,
        }
    else:
        qc = {
            'no_saccades': True,
        }
    
    if show_stats & len(valid_stats) > 0:
        mean_duration = np.nanmean(valid_stats['duration_ms'])
        mean_speed = np.nanmean(valid_stats['mean_speed'])
        max_speed = np.nanmean(valid_stats['max_speed'])
        print(
            f'rate = {trial_rate:.2f}, mean_duration = {mean_duration:.2f}ms, '
            f'mean_speed = {mean_speed:.2f}, max_speed = {max_speed:.2f}, '
            f'stdx = {stdx:.2f}, stdy = {stdy:.2f}, stdpup = {stdpup:.2f}, '
            f'blink% = {np.mean(blink_mask)*100:.1f}%'
        )
    return [reason for reason, flag in qc.items() if flag], valid_stats, trial_rate

def participant(participant, show_stats=True, final=True, time_window=None):
    file_path = get_control_file(participant)
    data = np.load(file_path)  # shape (channels, time, trials)
    n_trials = data.shape[2]
    fs = 200
    trial_len_s = data.shape[1] / fs

    # z-scoring across participant
    speed_mu, speed_sigma = _participant_speed_stats(data)

    rejected = 0
    rates = []
    agg = {'duration_ms': [], 'mean_speed': [], 'max_speed': []}
    fixation_rates = []
    fixation_durations_ms = []
    for trial_num in range(1, n_trials+1):
        if show_stats: print(f"\n=== Trial {trial_num} ===")
        qc, stats, _ = trial(
            participant,
            trial_num,
            show_plot=False,
            show_stats=False,
            final=final,
            speed_mu=speed_mu,
            speed_sigma=speed_sigma
        )
        if len(qc) > 0: 
            if show_stats: print(f"Trial {trial_num} rejected due to:", qc)
            rejected += 1
            continue
        else:
            if time_window is not None:
                start_s, end_s = time_window
                start_s = max(0.0, float(start_s))
                end_s = min(trial_len_s, float(end_s))
                window_len = end_s - start_s
                if window_len > 0:
                    stats = stats.loc[
                        (stats['start_s'] >= start_s) & (stats['end_s'] <= end_s)
                    ]
                    rate = len(stats) / window_len
                else:
                    rate = np.nan
            else:
                window_len = trial_len_s
                rate = len(stats) / trial_len_s

            rates.append(rate)
            agg['duration_ms'].extend(stats['duration_ms'])
            agg['mean_speed'].extend(stats['mean_speed'])
            agg['max_speed'].extend(stats['max_speed'])

            if window_len > 0 and len(stats) >= 2:
                stats_sorted = stats.sort_values('start_s')
                starts = stats_sorted['start_s'].to_numpy()
                ends = stats_sorted['end_s'].to_numpy()
                gaps_s = starts[1:] - ends[:-1]
                gaps_s = gaps_s[gaps_s >= 0]
            else:
                gaps_s = np.array([])

            fixation_rates.append(len(gaps_s) / window_len if window_len > 0 else np.nan)
            fixation_durations_ms.extend(gaps_s * 1000.0)

    if rates and agg['duration_ms']:
        mean_rate = np.mean(rates)
        mean_duration = np.nanmean(agg['duration_ms'])
        mean_mean_speed = np.nanmean(agg['mean_speed'])
        mean_max_speed = np.nanmean(agg['max_speed'])
        mean_fix_rate = np.nanmean(fixation_rates) if fixation_rates else np.nan
        mean_fix_duration = np.nanmean(fixation_durations_ms) if fixation_durations_ms else np.nan
        if show_stats:
            if time_window is not None:
                start_s, end_s = time_window
                window_label = f"{start_s}-{end_s}s"
                print(
                    f'\n[{window_label}] mean rate = {mean_rate:.2f}/sec, '
                    f'mean duration = {mean_duration:.2f}ms, '
                    f'avg mean speed = {mean_mean_speed:.2f}, '
                    f'avg max speed = {mean_max_speed:.2f}, '
                    f'fixation rate = {mean_fix_rate:.2f}/sec, '
                    f'mean fixation = {mean_fix_duration:.2f}ms'
                )
            else:
                print(
                    f'\nmean rate = {mean_rate:.2f}/sec, '
                    f'mean duration = {mean_duration:.2f}ms, '
                    f'avg mean speed = {mean_mean_speed:.2f}, '
                    f'avg max speed = {mean_max_speed:.2f}, '
                    f'fixation rate = {mean_fix_rate:.2f}/sec, '
                    f'mean fixation = {mean_fix_duration:.2f}ms'
                )
    else:
        mean_rate = mean_duration = mean_max_speed = mean_mean_speed = np.nan
        mean_fix_rate = mean_fix_duration = np.nan
        if show_stats:
            print("\nNo valid saccades across trials (all rejected or empty).")

    if show_stats:
        print(f"\nTrials rejected: {rejected} / {n_trials}")

    return mean_rate, mean_duration, mean_mean_speed, mean_max_speed, mean_fix_rate, mean_fix_duration
