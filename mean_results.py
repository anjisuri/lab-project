import os
import sf_con
import sf_pat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def con_means(show_plots=True):

    dir_path = Path(__file__).parent.resolve() / 'EyeData/controls'
    n = len(os.listdir(dir_path))

    rates = []
    durations = []
    mean_speeds = []
    max_speeds = []
    valid = []
    invalid = []

    for p in range(1, n + 1):
        mean_rate, mean_duration, mean_mean_speed, mean_max_speed = sf_con.participant(
            p, show_stats=False, final=False
        )
        if any(np.isnan(x) for x in [mean_rate, mean_duration, mean_mean_speed, mean_max_speed]):
            invalid.append(p)
            continue
        valid.append(p)
        rates.append(mean_rate)
        durations.append(mean_duration)
        mean_speeds.append(mean_mean_speed)
        max_speeds.append(mean_max_speed)

    if show_plots:
        plots = [
            ("Mean Rate", rates),
            ("Mean Duration", durations),
            ("Mean Speed", mean_speeds),
            ("Max Speed", max_speeds)
        ]

        for title, data in plots:
            plt.figure(figsize=(6,4))
            plt.plot(valid, data, marker="o")
            plt.title(title)
            plt.xlabel("Participant")
            plt.ylabel(title)
            for p in invalid:
                plt.axvline(p, ls='--', color = 'r', lw = '0.5')
            plt.tight_layout()
            plt.show()
    
    return {
        "rates": np.array(rates),
        "durations": np.array(durations),
        "mean_speeds": np.array(mean_speeds),
        "max_speeds": np.array(max_speeds),
    }

def pat_means(show_plots=True):

    dir_path = Path(__file__).parent.resolve() / 'EyeData/patients'
    n = len(os.listdir(dir_path))

    rates = []
    durations = []
    mean_speeds = []
    max_speeds = []
    valid = []
    invalid = []

    for p in range(1, n + 1):
        mean_rate, mean_duration, mean_mean_speed, mean_max_speed = sf_pat.participant(
            p, show_stats=False, final=False
        )
        if any(np.isnan(x) for x in [mean_rate, mean_duration, mean_mean_speed, mean_max_speed]):
            invalid.append(p)
            continue
        valid.append(p)
        rates.append(mean_rate)
        durations.append(mean_duration)
        mean_speeds.append(mean_mean_speed)
        max_speeds.append(mean_max_speed)

    if show_plots:
        plots = [
            ("Mean Rate", rates),
            ("Mean Duration", durations),
            ("Mean Speed", mean_speeds),
            ("Max Speed", max_speeds)
        ]

        for title, data in plots:
            plt.figure(figsize=(6,4))
            plt.plot(valid, data, marker="o")
            plt.title(title)
            plt.xlabel("Participant")
            plt.ylabel(title)
            for p in invalid:
                plt.axvline(p, ls='--', color = 'r', lw = '0.5')
            plt.tight_layout()
            plt.show()
    
    return {
        "rates": np.array(rates),
        "durations": np.array(durations),
        "mean_speeds": np.array(mean_speeds),
        "max_speeds": np.array(max_speeds),
    }
