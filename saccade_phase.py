import sf_con, sf_pat
from meg_filtering import hilbert
import matplotlib.pyplot as plt
import numpy as np

'''
- i want to find the time points when the saccades occur
- and then plot the phase at that point
- so first plot is saccades overlaying the phase
'''
def plot_saccades_over_phase(participant, trial, group='ctrl', hemi=0, fs=200):
    eye = sf_con if group == 'ctrl' else sf_pat
    qc, valid_stats, _ = eye.trial(participant, trial, show_plot=False, show_stats=False)

    out = hilbert(participant, trial, group=group, show_plots=False, fs=fs)
    phase = out['phase']

    t = np.arange(phase.shape[0]) / fs
    starts = valid_stats['start_idx'].to_numpy()
    ends = valid_stats['end_idx'].to_numpy()

    plt.figure(figsize=(12, 4))
    plt.plot(t, phase[:, hemi], label='phase')
    for i in range(len(starts)):
        plt.axvline(starts[i] / fs, color='g', lw=0.8, alpha=0.6)
        plt.axvline(ends[i] / fs, color='r', lw=0.8, alpha=0.6)
    plt.title(f'{group} p{participant} trial {trial} hemi {hemi}')
    plt.xlabel('time (s)')
    plt.ylabel('phase (rad)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_saccades_over_phase(1, 1, group='ctrl', hemi=0, fs=200)
