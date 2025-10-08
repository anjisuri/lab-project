from scipy.io import loadmat
import numpy as np
import pandas as pd

data = loadmat('/Users/anji/Desktop/lab project/eye_tracking_data/controls.mat', squeeze_me = True, struct_as_record = False)

eye = np.asarray(data['eyeTrck']).squeeze()

i = 1
for subject in eye:
    if subject.ndim == 3:
        reshaped_data = np.transpose(subject, (1,2,0)).reshape(-1,3) # reorders axes (from channels, timepoints, trials to timepoints, trials, channels) 
        df = pd.DataFrame (reshaped_data, columns = ['x', 'y', 'pupil'])
        df.to_csv(f'/Users/anji/Desktop/lab project/python_data/controls/participant_{i}.csv', index = False)
        i += 1
    else:
        print(f'Skipping row {i} (empty data)')
        continue