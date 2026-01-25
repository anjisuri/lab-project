import sf_con, sf_pat
from meg_filtering import hilbert
import matplotlib.pyplot as plt
import numpy as np
from config import get_control_file, get_patient_file

'''
- i want to find the time points when the saccades occur
- and then plot the phase at that point
- so first plot is saccades overlaying the phase
'''

def sp(participant, trial, group='ctrl', hemi=0, fs=200, show_plot=True):
    eye = sf_con if group == 'ctrl' else sf_pat
    qc, valid_stats, _ = eye.trial(participant, trial, show_plot=False, show_stats=False)

    out = hilbert(participant, trial, group=group, show_plots=False, fs=fs)
    phase = out['phase']

    t = np.arange(phase.shape[0]) / fs
    starts = valid_stats['start_idx'].to_numpy()
    ends = valid_stats['end_idx'].to_numpy()

    phase_at_starts = phase[starts, hemi]
    phase_at_ends = phase[ends, hemi]

    if show_plot:
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
    
    return phase_at_starts, phase_at_ends

def phase_dist(participant, group='ctrl', hemi=0, fs=200, which='start', bins=30, show_plot=True):
    file_path = get_control_file(participant) if group == 'ctrl' else get_patient_file(participant)
    data = np.load(file_path)
    n_trials = data.shape[2]

    phases = []
    for trial in range(1, n_trials + 1):
        start_phase, end_phase = sp(
            participant,
            trial,
            group=group,
            hemi=hemi,
            fs=fs,
            show_plot=False,
        )
        if which == 'start':
            phases.extend(start_phase)
        elif which == 'end':
            phases.extend(end_phase)
        else:
            phases.extend(start_phase)
            phases.extend(end_phase)

    phases = np.asarray(phases)

    if show_plot:
        plt.figure(figsize=(6, 4))
        plt.hist(phases, bins=bins, density=True)
        plt.title(f'Phase distribution: {group} p{participant} hemi {hemi} ({which})')
        plt.xlabel('phase (rad)')
        plt.ylabel('density')
        plt.tight_layout()
        plt.show()

    return phases

def phase_dist_group(group='ctrl', hemi=0, fs=200, which='start', bins=30, show_plot=True, per_participant_plot=False):
    last = 26 if group == 'ctrl' else 18
    all_phases = []
    for participant in range(1, last + 1):
        if group == 'ctrl' and participant == 6:
            continue
        phases = phase_dist(
            participant,
            group=group,
            hemi=hemi,
            fs=fs,
            which=which,
            bins=bins,
            show_plot=per_participant_plot,
        )
        if phases is not None and len(phases) > 0:
            all_phases.extend(phases)


    if show_plot:
        plt.figure(figsize=(6, 4))
        plt.hist(all_phases, bins=bins, density=True)
        plt.title(f'Phase distribution: {group} hemi {hemi} ({which})')
        plt.xlabel('phase (rad)')
        plt.ylabel('density')
        plt.tight_layout()
        plt.show()

    return all_phases
