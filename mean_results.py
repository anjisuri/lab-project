import os
import sf_con
import sf_pat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from config import list_control_ids, list_patient_ids

def con_means(show_plots=True):
    control_ids = list_control_ids(exclude=True)

    rates = []
    durations = []
    mean_speeds = []
    max_speeds = []
    fixation_rates = []
    fixation_durations = []
    valid = []
    invalid = []

    for p in control_ids:
        mean_rate, mean_duration, mean_mean_speed, mean_max_speed, mean_fix_rate, mean_fix_duration = sf_con.participant(
            p, show_stats=False, final=False
        )
        if all(np.isnan(x) for x in [mean_rate, mean_duration, mean_mean_speed, mean_max_speed, mean_fix_rate, mean_fix_duration]):
            invalid.append(p)
            continue
        valid.append(p)
        rates.append(mean_rate)
        durations.append(mean_duration)
        mean_speeds.append(mean_mean_speed)
        max_speeds.append(mean_max_speed)
        fixation_rates.append(mean_fix_rate)
        fixation_durations.append(mean_fix_duration)

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

        fixation_plots = [
            ("Fixation Rate", fixation_rates),
            ("Fixation Duration", fixation_durations),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (title, data) in zip(axes.ravel(), fixation_plots):
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
        "fixation_rates": np.array(fixation_rates),
        "fixation_durations": np.array(fixation_durations),
    }

def pat_means(show_plots=True):
    patient_ids = list_patient_ids()

    rates = []
    durations = []
    mean_speeds = []
    max_speeds = []
    fixation_rates = []
    fixation_durations = []
    valid = []
    invalid = []

    for p in patient_ids:
        mean_rate, mean_duration, mean_mean_speed, mean_max_speed, mean_fix_rate, mean_fix_duration = sf_pat.participant(
            p, show_stats=False, final=False
        )
        if all(np.isnan(x) for x in [mean_rate, mean_duration, mean_mean_speed, mean_max_speed, mean_fix_rate, mean_fix_duration]):
            invalid.append(p)
            continue
        valid.append(p)
        rates.append(mean_rate)
        durations.append(mean_duration)
        mean_speeds.append(mean_mean_speed)
        max_speeds.append(mean_max_speed)
        fixation_rates.append(mean_fix_rate)
        fixation_durations.append(mean_fix_duration)

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

        fixation_plots = [
            ("Fixation Rate", fixation_rates),
            ("Fixation Duration", fixation_durations),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (title, data) in zip(axes.ravel(), fixation_plots):
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
        "fixation_rates": np.array(fixation_rates),
        "fixation_durations": np.array(fixation_durations),
    }

def _mean_or_nan(values):
    if not values:
        return np.nan
    return float(np.nanmean(values))

def _finite_count(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0
    return int(np.isfinite(arr).sum())

def con_means_window(start_s, end_s, show_plots=True, return_average=True):
    control_ids = list_control_ids(exclude=True)

    rates = []
    durations = []
    mean_speeds = []
    max_speeds = []
    fixation_rates = []
    fixation_durations = []
    valid = []
    invalid = []
    window_label = f"{start_s}-{end_s}s"

    for p in control_ids:
        mean_rate, mean_duration, mean_mean_speed, mean_max_speed, mean_fix_rate, mean_fix_duration = sf_con.participant(
            p, show_stats=False, final=False, time_window=(start_s, end_s)
        )
        if all(np.isnan(x) for x in [mean_rate, mean_duration, mean_mean_speed, mean_max_speed, mean_fix_rate, mean_fix_duration]):
            invalid.append(p)
            continue
        valid.append(p)
        rates.append(mean_rate)
        durations.append(mean_duration)
        mean_speeds.append(mean_mean_speed)
        max_speeds.append(mean_max_speed)
        fixation_rates.append(mean_fix_rate)
        fixation_durations.append(mean_fix_duration)

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

        fixation_plots = [
            (f"Fixation Rate ({window_label})", fixation_rates),
            (f"Fixation Duration ({window_label})", fixation_durations),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (title, data) in zip(axes.ravel(), fixation_plots):
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
            "fixation_rates": _mean_or_nan(fixation_rates),
            "fixation_durations": _mean_or_nan(fixation_durations),
            "n_valid": _finite_count(rates),
            "valid_ids": np.array(valid, dtype=int),
        }

    return {
        "rates": np.array(rates),
        "durations": np.array(durations),
        "mean_speeds": np.array(mean_speeds),
        "max_speeds": np.array(max_speeds),
        "fixation_rates": np.array(fixation_rates),
        "fixation_durations": np.array(fixation_durations),
        "valid_ids": np.array(valid, dtype=int),
    }

def pat_means_window(start_s, end_s, show_plots=True, return_average=True):
    patient_ids = list_patient_ids()

    rates = []
    durations = []
    mean_speeds = []
    max_speeds = []
    fixation_rates = []
    fixation_durations = []
    valid = []
    invalid = []
    window_label = f"{start_s}-{end_s}s"

    for p in patient_ids:
        mean_rate, mean_duration, mean_mean_speed, mean_max_speed, mean_fix_rate, mean_fix_duration = sf_pat.participant(
            p, show_stats=False, final=False, time_window=(start_s, end_s)
        )
        if all(np.isnan(x) for x in [mean_rate, mean_duration, mean_mean_speed, mean_max_speed, mean_fix_rate, mean_fix_duration]):
            invalid.append(p)
            continue
        valid.append(p)
        rates.append(mean_rate)
        durations.append(mean_duration)
        mean_speeds.append(mean_mean_speed)
        max_speeds.append(mean_max_speed)
        fixation_rates.append(mean_fix_rate)
        fixation_durations.append(mean_fix_duration)

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

        fixation_plots = [
            (f"Fixation Rate ({window_label})", fixation_rates),
            (f"Fixation Duration ({window_label})", fixation_durations),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (title, data) in zip(axes.ravel(), fixation_plots):
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
            "fixation_rates": _mean_or_nan(fixation_rates),
            "fixation_durations": _mean_or_nan(fixation_durations),
            "n_valid": _finite_count(rates),
            "valid_ids": np.array(valid, dtype=int),
        }

    return {
        "rates": np.array(rates),
        "durations": np.array(durations),
        "mean_speeds": np.array(mean_speeds),
        "max_speeds": np.array(max_speeds),
        "fixation_rates": np.array(fixation_rates),
        "fixation_durations": np.array(fixation_durations),
        "valid_ids": np.array(valid, dtype=int),
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
