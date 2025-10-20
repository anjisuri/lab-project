import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import zscore
from skimage import measure

def analyse(participant, thr_z, merging_ms):
    df = pd.read_csv(f'/Users/anji/Desktop/lab project/python_data/controls/participant_{participant}.csv')
    fs = 200 #sampling frequency, Hz

    samples = 1601
    trials = df.shape[0] // samples
    df['trial'] = np.repeat(np.arange(1, trials + 1), samples)
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

        plt.figure(figsize=(10,4))

        plt.plot(time, speed_z)


        plt.title('Instantaneous eye movement speed')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed, zscored (a.u.)')
        plt.tight_layout()

        # --- saccade statistics ---
        iters = max(1, int((merging_ms / 1000.0) * fs))   # samples to merge at fs Hz

        raw = speed_z > thr_z
        print('raw > thr_z:', int(raw.sum()))
        print('iters (samples merged):', iters)
        # merge short gaps between supra-threshold samples
        saccades = ndimage.binary_dilation(raw, iterations=iters)
        # optional gentle cleanup to remove isolated single-sample noise (commented by default)
        # saccades = ndimage.binary_opening(saccades)

        labelled, n = ndimage.label(saccades.astype(np.uint8))
        regions = ndimage.find_objects(labelled)

        for i in regions: 
            start = (time[i[0].start]).round(4)
            end = (time[i[0].stop - 1]).round(4)
            duration_ms = ((end - start)* 1000).round(4)
            print(f'{start}s -> {end}s  (duration {duration_ms}ms)')

        movement = saccades
        print(f'number of saccades = {n}')

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

        stats_df['end_idx'] = stats_df['end_idx_exclusive'] - 1
        stats_df['start_s'] = stats_df['start_idx'] / fs
        stats_df['end_s'] = stats_df['end_idx'] / fs
        stats_df['duration_ms'] = (stats_df['n_samples'] / fs) * 1000


        print(f"number of saccades = {len(stats_df)}")
        print(stats_df[['label','start_s','end_s','duration_ms','mean_speed','max_speed']])