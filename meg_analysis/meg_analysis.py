import mne
import matplotlib.pyplot as plt
import numpy as np
from autoreject import get_rejection_threshold
import os.path as op


# --- loading ---
epo_path = '/Users/anji/Desktop/lab project/meg_analysis/Participant5/effMpStudyData_epo.fif'
epochs = mne.read_epochs(epo_path, preload = True)

# --- sanity checks ---

# print(epochs.get_data().shape)
# print(np.nanmin(epochs.get_data()), np.nanmax(epochs.get_data()))

# print(epochs.info['ch_names'][:10])
# print(epochs.info['bads'])

# info = mne.io.read_info(epo_path)
# print(info.keys())
# print(info['ch_names'])

# --- plotting ---

chs = [
    'MLC11', 'MLC12', 'MLC13', 
    'MLC14', 'MLC15', 'MLC16', 
    'MLC17', 'MLC21', 'MLC22'
]

# epochs.compute_psd(fmax=50).plot(picks = chs, exclude = 'bads', amplitude = False)
# epochs.plot(scalings='auto', n_channels=5)
# epochs['cond_1'].plot(scalings = 'auto', picks=chs)
# plt.show()

# --- remove noise and drop bad epochs ---
ep = epochs.copy()
ep.filter(1., 40., picks='data', fir_design='firwin')

# 1) auto-compute a sensible reject dict for gradiometers
reject = get_rejection_threshold(ep, ch_types='grad')
print("Reject dict:", reject)

# 2) drop bad epochs using that threshold
ep_clean = ep.copy().drop_bad(reject=reject)

# 3) plot cleaned data
ep_clean['cond_1'].plot(scalings='auto', picks=chs)
plt.show()