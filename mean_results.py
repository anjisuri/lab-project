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
            ("Max Speed", max_speeds),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (title, data) in zip(axes.ravel(), plots):
            ax.plot(valid, data, marker="o")
            ax.set_title(title)
            ax.set_xlabel("Participant")
            ax.set_ylabel(title)
            for p in invalid:
                ax.axvline(p, ls="--", color="r", lw="0.5")

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
            ("Max Speed", max_speeds),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (title, data) in zip(axes.ravel(), plots):
            ax.plot(valid, data, marker="o")
            ax.set_title(title)
            ax.set_xlabel("Participant")
            ax.set_ylabel(title)
            for p in invalid:
                ax.axvline(p, ls="--", color="r", lw="0.5")

        plt.tight_layout()
        plt.show()
    
    return {
        "rates": np.array(rates),
        "durations": np.array(durations),
        "mean_speeds": np.array(mean_speeds),
        "max_speeds": np.array(max_speeds),
    }

def _mean_or_nan(values):
    if not values:
        return np.nan
    return float(np.nanmean(values))

def con_means_window(start_s, end_s, show_plots=True, return_average=True):
    dir_path = Path(__file__).parent.resolve() / 'EyeData/controls'
    n = len(os.listdir(dir_path))

    rates = []
    durations = []
    mean_speeds = []
    max_speeds = []
    valid = []
    invalid = []
    window_label = f"{start_s}-{end_s}s"

    for p in range(1, n + 1):
        mean_rate, mean_duration, mean_mean_speed, mean_max_speed = sf_con.participant(
            p, show_stats=False, final=False, time_window=(start_s, end_s)
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
            (f"Mean Rate ({window_label})", rates),
            (f"Mean Duration ({window_label})", durations),
            (f"Mean Speed ({window_label})", mean_speeds),
            (f"Max Speed ({window_label})", max_speeds),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (title, data) in zip(axes.ravel(), plots):
            ax.plot(valid, data, marker="o")
            ax.set_title(title)
            ax.set_xlabel("Participant")
            ax.set_ylabel(title)
            for p in invalid:
                ax.axvline(p, ls="--", color="r", lw="0.5")

        plt.tight_layout()
        plt.show()

    if return_average:
        return {
            "rates": _mean_or_nan(rates),
            "durations": _mean_or_nan(durations),
            "mean_speeds": _mean_or_nan(mean_speeds),
            "max_speeds": _mean_or_nan(max_speeds),
            "n_valid": len(rates),
        }

    return {
        "rates": np.array(rates),
        "durations": np.array(durations),
        "mean_speeds": np.array(mean_speeds),
        "max_speeds": np.array(max_speeds),
    }

def pat_means_window(start_s, end_s, show_plots=True, return_average=True):
    dir_path = Path(__file__).parent.resolve() / 'EyeData/patients'
    n = len(os.listdir(dir_path))

    rates = []
    durations = []
    mean_speeds = []
    max_speeds = []
    valid = []
    invalid = []
    window_label = f"{start_s}-{end_s}s"

    for p in range(1, n + 1):
        mean_rate, mean_duration, mean_mean_speed, mean_max_speed = sf_pat.participant(
            p, show_stats=False, final=False, time_window=(start_s, end_s)
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
            (f"Mean Rate ({window_label})", rates),
            (f"Mean Duration ({window_label})", durations),
            (f"Mean Speed ({window_label})", mean_speeds),
            (f"Max Speed ({window_label})", max_speeds),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (title, data) in zip(axes.ravel(), plots):
            ax.plot(valid, data, marker="o")
            ax.set_title(title)
            ax.set_xlabel("Participant")
            ax.set_ylabel(title)
            for p in invalid:
                ax.axvline(p, ls="--", color="r", lw="0.5")

        plt.tight_layout()
        plt.show()

    if return_average:
        return {
            "rates": _mean_or_nan(rates),
            "durations": _mean_or_nan(durations),
            "mean_speeds": _mean_or_nan(mean_speeds),
            "max_speeds": _mean_or_nan(max_speeds),
            "n_valid": len(rates),
        }

    return {
        "rates": np.array(rates),
        "durations": np.array(durations),
        "mean_speeds": np.array(mean_speeds),
        "max_speeds": np.array(max_speeds),
    }

def window_means(windows=((1, 4), (4, 7)), show_plots=False):
    controls = {}
    patients = {}

    for start_s, end_s in windows:
        window_key = f"{start_s}-{end_s}s"
        controls[window_key] = con_means_window(
            start_s, end_s, show_plots=show_plots, return_average=True
        )
        patients[window_key] = pat_means_window(
            start_s, end_s, show_plots=show_plots, return_average=True
        )

    return {
        "controls": controls,
        "patients": patients,
    }

def hists(show_plots=True, bins=20):
    controls = con_means(show_plots=False)
    patients = pat_means(show_plots=False)

    if show_plots:
        plots = [
            ("Mean Rate", "rates"),
            ("Mean Duration", "durations"),
            ("Mean Speed", "mean_speeds"),
            ("Max Speed", "max_speeds"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (title, key) in zip(axes.ravel(), plots):
            ax.hist(controls[key], bins=bins, alpha=0.6, label="Controls")
            ax.hist(patients[key], bins=bins, alpha=0.6, label="Patients")
            ax.set_title(title)
            ax.set_xlabel(title)
            ax.set_ylabel("Count")
            ax.legend()

        plt.tight_layout()
        plt.show()

    return {
        "controls": controls,
        "patients": patients,
    }