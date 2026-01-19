import numpy as np
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
from config import get_meg_ctrl, get_meg_pat



def theta_filter(data, fs=200, band=(1, 8), order=5, axis=1, method='gust'):
    b, a = signal.butter(order, band, btype='bandpass', fs=fs)
    return signal.filtfilt(b, a, data, axis=axis, padlen=0)


def filter_participant(participant, group='ctrl', fs=200, band=(1, 8), order=5):
    file_path = get_meg_ctrl(participant) if group == 'ctrl' else get_meg_pat(participant)
    data = np.load(file_path)  # shape: (trials, time, hemisphere)
    return theta_filter(data, fs=fs, band=band, order=order)


def filter_trial(participant, trial, group='ctrl', fs=200, band=(1, 8), order=5):
    file_path = get_meg_ctrl(participant) if group == 'ctrl' else get_meg_pat(participant)
    data = np.load(file_path)  # shape: (trials, time, hemisphere)
    trial_data = data[trial - 1]
    return theta_filter(trial_data, fs=fs, band=band, order=order, axis=0)


def filter_all(group='ctrl', fs=200, band=(1, 8), order=5):
    example_path = get_meg_ctrl(1) if group == 'ctrl' else get_meg_pat(1)
    dir_path = Path(example_path).parent
    results = {}
    for path in sorted(dir_path.glob('*.npy')):
        data = np.load(path)
        results[path.stem] = theta_filter(data, fs=fs, band=band, order=order)
    return results


def plot_raw_vs_filtered(participant, trial, group='ctrl', fs=200, band=(1, 8), order=5):
    file_path = get_meg_ctrl(participant) if group == 'ctrl' else get_meg_pat(participant)
    data = np.load(file_path)  # shape: (trials, time, hemisphere)
    raw = data[trial - 1]
    filt = theta_filter(raw, fs=fs, band=band, order=order, axis=0)

    t = np.arange(raw.shape[0]) / fs
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

    for i in (0,1):
        side = 'left' if i == 0 else 'right'
        axes[i].plot(t, raw[:, i], label=f'{side} raw')
        axes[i].plot(t, filt[:, i], label=f'{side} filtered')
        axes[i].set_title(f'{group} p{participant} trial {trial} {side}')
        axes[i].set_ylabel('signal')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def hilbert(participant, trial, group='ctrl', fs=200, band=(1, 8), order=5, show_plots=True):
    file_path = get_meg_ctrl(participant) if group == 'ctrl' else get_meg_pat(participant)
    data = np.load(file_path)  # shape: (trials, time, hemisphere)
    raw = data[trial - 1]
    filt = theta_filter(raw, fs=fs, band=band, order=order, axis=0)
    analytic = signal.hilbert(filt, axis=0)
    envelope = np.abs(analytic)
    phase = np.angle(analytic)
    t = np.arange(raw.shape[0]) / fs

    if show_plots:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
        for i in (0, 1):
            side = 'left' if i == 0 else 'right'
            axes[i].plot(t, filt[:, i], label=f'{side} filtered')
            axes[i].plot(t, envelope[:, i], label=f'{side} envelope')
            axes[i].set_title(f'{group} p{participant} trial {trial} {side}')
            axes[i].set_ylabel('signal')
            axes[i].legend()
        axes[-1].set_xlabel('time (s)')
        plt.tight_layout()
        plt.show()

    return {
        'filtered': filt,
        'analytic': analytic,
        'envelope': envelope,
        'phase': phase,
    }
