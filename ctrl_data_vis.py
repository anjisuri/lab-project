import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from skimage import measure

fs = 200 #sampling frequency, Hz

# for i in range(1,2):
# df = pd.read_csv(f'/Users/anji/Desktop/lab project/python_data/controls/participant_{i}.csv')

x = 1
trial = 58
df = pd.read_csv(f'/Users/anji/Desktop/lab project/python_data/controls/participant_{x}.csv')

# plotting specific trial

samples = 1601
trials = df.shape[0] // samples
df['trial'] = np.repeat(np.arange(1, trials + 1), samples)

trial = df[df['trial'] == trial]
x = trial['x']
y = trial['y']
pupil = trial['pupil']

xdiff = np.diff(x) ** 2
ydiff = np.diff(y) ** 2
speed = np.sqrt(xdiff + ydiff)
speed_z = stats.zscore(speed)
time = np.arange(len(speed)) / fs  # seconds (fs = 200Hz)

plt.figure(figsize=(10,4))

plt.plot(time, speed_z)


plt.title('Instantaneous eye movement speed')
plt.xlabel('Time (s)')
plt.ylabel('Speed, zscored (a.u.)')
plt.tight_layout()
# plt.show()

# saccade statistics (pre-erosion)
# binary array of movement vs not movement

# threshold = 3
# saccades = ndimage.binary_dilation(speed_z > threshold, iterations = 1)   # each iteration 5ms merged
# saccades = ndimage.binary_erosion(saccades)
# labelled, n = ndimage.label(saccades)

# --- saccade statistics (no post-erosion collapse) ---
thr_z = 1.5                      # threshold in z-score units
merge_ms = 10                  # merge stationary gaps shorter than this
iters = max(1, int((merge_ms / 1000.0) * fs))   # samples to merge at fs Hz

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

stats = pd.DataFrame(props).rename(
    columns = {
        'area': 'n_samples',
        'bbox-0': 'start_idx',
        'bbox-2': 'end_idx_exclusive',
        'mean_intensity': 'mean_speed',
        'max_intensity': 'max_speed'
    }
)

stats['end_idx'] = stats['end_idx_exclusive'] - 1
stats['start_s'] = stats['start_idx'] / fs
stats['end_s'] = stats['end_idx'] / fs
stats['duration_ms'] = (stats['n_samples'] / fs) * 1000


print(f"number of saccades = {len(stats)}")
print(stats[['label','start_s','end_s','duration_ms','mean_speed','max_speed']])