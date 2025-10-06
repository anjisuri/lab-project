from scipy.io import loadmat
import numpy as np
import pandas as pd

data = loadmat('/Users/anji/Desktop/lab project/eye_tracking_data/patients.mat', squeeze_me = True, struct_as_record = False)

eye = np.asarray(data['eyeTrck']).squeeze()

for i, subject in enumerate(eye, start = 1):
    if subject.ndim == 3:
        reshaped_data = np.transpose(subject, (1,2,0)).reshape(-1,3) # reorders axes (from channels, timepoints, trials to timepoints, trials, channels) 
        df = pd.DataFrame (reshaped_data, columns = ['x', 'y', 'pupil'])
        df.to_csv(f'/Users/anji/Desktop/lab project/python_data/patients/participant_{i}.csv', index = False)
    else:
        print(f'Skipping row {i} (empty data) / wrong shape')
        continue




# xdiff = np.diff(trial1[0, :]) ** 2
# ydiff = np.diff(trial1[1,:]) ** 2
# speed = np.sqrt(xdiff + ydiff)
# plt.figure(figsize=(10,4))
# time = np.arange(len(speed)) / 200  # seconds
# plt.plot(time, speed)
# plt.title('Instantaneous eye movement speed (Trial 1)')
# plt.xlabel('Time (s)')
# plt.ylabel('Speed (a.u.)')
# plt.tight_layout()
# plt.show()