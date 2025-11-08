# George O'Neill 2023 (adapted)
import mne
from osl_ephys.utils.spmio import SPMMEEG
import numpy as np
import os.path as op

# --- Input file (.mat must sit next to its .dat) ---
fname = op.abspath("/Users/anji/Desktop/lab project/meg_analysis/Participant5/effMpStudyData.mat")

# --- Load and fix internal .dat pointer (Windows paths inside MAT can break on macOS) ---
D = SPMMEEG(fname)
_dat_local = op.splitext(fname)[0] + ".dat"
if hasattr(D, "datname"):
    try:
        D.datname = _dat_local
    except Exception:
        pass
if hasattr(D, "_spm_data"):
    try:
        D._spm_data["data"]["fname"] = _dat_local
    except Exception:
        pass

# --- Basic info ---
fs = D.fsample
ch_names = [ch.label for ch in D.channels]
ch_types = [ct for ct in D.chantype]
# map to MNE channel types
ch_types = ["grad" if x == "MEGGRAD" else x for x in ch_types]
ch_types = ["ref_meg" if (x == "REFGRAD" or x == "REFMAG") else x for x in ch_types]
ch_types = ["misc" if x == "Other" else x for x in ch_types]
info = mne.create_info(ch_names, fs, ch_types=ch_types)

# --- Data to (trials, channels, samples) ---
dat = D.get_data()
dat = np.rollaxis(dat, -1)

# --- Build events ---
'''
We need one event per trial. Choose:
- marker = 'button' : rising edge in UPPT002 inside each trial
- marker = 'epoch'  : start of each trial

First column is the sample index in the continuous stream:
base for trial i is i * D.nsamples
'''
marker = 'button'  # or 'epoch'
events = np.zeros((D.ntrials, 3), dtype=int)
# condition codes provided by the dataset
events[:, 2] = np.asarray(D.conditions, dtype=int)

if marker == 'button':
    print('Using button presses as event marker')
    try:
        cid = ch_names.index('UPPT002')
    except ValueError:
        raise RuntimeError("Channel 'UPPT002' not found in ch_names")
    trig_data = dat[:, cid, :]  # (trials, samples)
    for ii in range(D.ntrials):
        base = ii * int(D.nsamples)
        dd = np.hstack([0, np.diff(trig_data[ii])])   # rising edge
        onset = int(dd.argmax())
        events[ii, 0] = base + onset
elif marker == 'epoch':
    print('Using start of epoch as event marker')
    events[:, 0] = np.arange(D.ntrials, dtype=int) * int(D.nsamples)
else:
    raise ValueError(f"Unknown marker: {marker}")

# --- Dynamic event_id from present codes ---
present_codes = np.unique(events[:, 2]).astype(int)
if present_codes.size == 0:
    raise RuntimeError("No events present - check marker detection.")
event_id = {f"cond_{c}": int(c) for c in present_codes}
print("Event codes found:", present_codes, "-> event_id:", event_id)

# --- Create epochs and save ---
epochs = mne.EpochsArray(dat, info=info, events=events, event_id=event_id)
epochs.save(op.splitext(fname)[0] + '_epo.fif')