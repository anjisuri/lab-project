import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure


# for i in range(1,2):
#df = pd.read_csv(f'/Users/anji/Desktop/lab project/python_data/controls/participant_{i}.csv')
df = pd.read_csv('/Users/anji/Desktop/lab project/python_data/controls/participant_1.csv')

# plotting trial 1 

samples = 1601
trials = df.shape[0] // samples
df['trial'] = np.repeat(np.arange(1, trials + 1), samples)

trial1 = df[df['trial'] == 1]
x = trial1['x']
y = trial1['y']
pupil = trial1['pupil']

xdiff = np.diff(x) ** 2
ydiff = np.diff(y) ** 2
speed = np.sqrt(xdiff + ydiff)
time = np.arange(len(speed)) / 200  # seconds

plt.figure(figsize=(10,4))

plt.plot(time, speed)

plt.xlim(0,2)
plt.ylim(0)

plt.title('Instantaneous eye movement speed')
plt.xlabel('Time (s)')
plt.ylabel('Speed (a.u.)')
plt.tight_layout()

# saccade statistics (pre-erosion)
threshold = 7
saccades = ndimage.binary_dilation(speed > threshold, iterations = 6)
saccades = ndimage.binary_erosion(saccades)
labelled, n = ndimage.label(saccades)
regions = ndimage.find_objects(labelled)

for i in regions:
    start = (time[i[0].start]).round(4)
    end = (time[i[0].stop - 1]).round(4)
    duration_ms = ((end - start)* 1000).round(4)
    print(f'{start}s -> {end}s  (duration {duration_ms}ms)')

movement = saccades
print(f'number of saccades = {n}')

plt.plot(time,movement * max(speed))

plt.show()
