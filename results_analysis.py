import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from mean_results import con_means, pat_means, con_means_window, pat_means_window
from rayleigh import rayleigh_window_group_df

plt.rcParams["font.family"] = "Helvetica"
OUT_DIR = "analysis_outputs"


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(x.size)


def summarize_overall():
    c = con_means(show_plots=False)
    p = pat_means(show_plots=False)

    all_sacc_rate = np.concatenate([c["rates"], p["rates"]])
    all_sacc_dur_s = np.concatenate([c["durations"], p["durations"]]) / 1000.0
    all_fix_rate = np.concatenate([c["fixation_rates"], p["fixation_rates"]])
    all_fix_dur_s = np.concatenate([c["fixation_durations"], p["fixation_durations"]]) / 1000.0

    return {
        "n_controls": int(np.isfinite(c["rates"]).sum()),
        "n_patients": int(np.isfinite(p["rates"]).sum()),
        "n_total": int(np.isfinite(all_sacc_rate).sum()),
        "saccade_rate_hz_mean": float(np.nanmean(all_sacc_rate)),
        "saccade_rate_hz_sem": float(sem(all_sacc_rate)),
        "saccade_dur_s_mean": float(np.nanmean(all_sacc_dur_s)),
        "saccade_dur_s_sem": float(sem(all_sacc_dur_s)),
        "fix_rate_hz_mean": float(np.nanmean(all_fix_rate)),
        "fix_rate_hz_sem": float(sem(all_fix_rate)),
        "fix_dur_s_mean": float(np.nanmean(all_fix_dur_s)),
        "fix_dur_s_sem": float(sem(all_fix_dur_s)),
    }


def window_metric_dfs():
    con_fix = con_means_window(1, 4, show_plots=False, return_average=False)
    con_cue = con_means_window(4, 7, show_plots=False, return_average=False)
    pat_fix = pat_means_window(1, 4, show_plots=False, return_average=False)
    pat_cue = pat_means_window(4, 7, show_plots=False, return_average=False)

    def make_df(key):
        rows = []
        for i, (a, b) in enumerate(zip(con_fix[key], con_cue[key]), start=1):
            if np.isfinite(a) and np.isfinite(b):
                rows.append({"subject": f"C{i:02d}", "group": "control", "fixation": float(a), "cue": float(b)})
        for i, (a, b) in enumerate(zip(pat_fix[key], pat_cue[key]), start=1):
            if np.isfinite(a) and np.isfinite(b):
                rows.append({"subject": f"P{i:02d}", "group": "patient", "fixation": float(a), "cue": float(b)})
        return pd.DataFrame(rows)

    return {
        "saccade_frequency": make_df("rates"),
        "saccade_duration_ms": make_df("durations"),
        "mean_saccade_speed": make_df("mean_speeds"),
        "max_saccade_speed": make_df("max_speeds"),
        "fixation_frequency": make_df("fixation_rates"),
        "fixation_duration_ms": make_df("fixation_durations"),
    }


def phase_metric_df(which):
    c = rayleigh_window_group_df(group="ctrl", which=which, windows=((1, 4), (4, 7)), final=False)
    p = rayleigh_window_group_df(group="patient", which=which, windows=((1, 4), (4, 7)), final=False)
    c = c.rename(columns={"participant": "id", "r": "value"}).copy()
    p = p.rename(columns={"participant": "id", "r": "value"}).copy()
    c["group"] = "control"
    p["group"] = "patient"
    c["subject"] = c["id"].astype(int).map(lambda x: f"C{x:02d}")
    p["subject"] = p["id"].astype(int).map(lambda x: f"P{x:02d}")

    df = pd.concat([c, p], ignore_index=True)
    df["condition"] = list(zip(df["window_start_s"], df["window_end_s"]))
    df["condition"] = df["condition"].map({(1, 4): "fixation", (4, 7): "cue"})

    wide = df.pivot_table(index=["subject", "group"], columns="condition", values="value", aggfunc="mean")
    wide = wide.dropna().reset_index()
    return wide[["subject", "group", "fixation", "cue"]]


def split_plot_anova_equivalent(wide_df):
    df = wide_df.copy()
    df["mean"] = (df["fixation"] + df["cue"]) / 2.0
    df["diff"] = df["cue"] - df["fixation"]

    g_control = df[df["group"] == "control"]
    g_patient = df[df["group"] == "patient"]

    # Group main effect on subject means
    t_group = stats.ttest_ind(g_control["mean"], g_patient["mean"], equal_var=False, nan_policy="omit")
    n1, n2 = len(g_control), len(g_patient)
    s1 = np.var(g_control["mean"], ddof=1)
    s2 = np.var(g_patient["mean"], ddof=1)
    df_group = (s1 / n1 + s2 / n2) ** 2 / ((s1**2) / (n1**2 * (n1 - 1)) + (s2**2) / (n2**2 * (n2 - 1)))

    # Condition main effect on within-subject differences across all subjects
    t_cond = stats.ttest_1samp(df["diff"], popmean=0.0, nan_policy="omit")
    df_cond = len(df) - 1

    # Interaction effect: group difference in within-subject differences
    t_int = stats.ttest_ind(g_control["diff"], g_patient["diff"], equal_var=False, nan_policy="omit")
    s1d = np.var(g_control["diff"], ddof=1)
    s2d = np.var(g_patient["diff"], ddof=1)
    df_int = (s1d / n1 + s2d / n2) ** 2 / ((s1d**2) / (n1**2 * (n1 - 1)) + (s2d**2) / (n2**2 * (n2 - 1)))

    out = {
        "group": {"F": float(t_group.statistic**2), "df1": 1, "df2": float(df_group), "p": float(t_group.pvalue)},
        "condition": {"F": float(t_cond.statistic**2), "df1": 1, "df2": float(df_cond), "p": float(t_cond.pvalue)},
        "interaction": {"F": float(t_int.statistic**2), "df1": 1, "df2": float(df_int), "p": float(t_int.pvalue)},
        "n_control": int(n1),
        "n_patient": int(n2),
        "cell_stats": cell_stats(wide_df),
    }
    return out


def cell_stats(wide_df):
    rows = []
    for group in ["control", "patient"]:
        sub = wide_df[wide_df["group"] == group]
        for cond in ["fixation", "cue"]:
            vals = sub[cond].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            rows.append(
                {
                    "group": group,
                    "condition": cond,
                    "mean": float(np.mean(vals)),
                    "sem": float(sem(vals)),
                    "count": int(vals.size),
                    "std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_cells(cell_df, title, ylabel, outfile):
    groups = ["control", "patient"]
    conds = ["fixation", "cue"]
    x = np.arange(len(groups)) * 0.45
    width = 0.12

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for j, cond in enumerate(conds):
        means, errs = [], []
        for g in groups:
            row = cell_df[(cell_df["group"] == g) & (cell_df["condition"] == cond)].iloc[0]
            means.append(row["mean"])
            errs.append(row["sem"])
        pos = x + (j - 0.5) * width
        ax.bar(pos, means, width=width, alpha=0.9, label=cond.capitalize())
        ax.errorbar(pos, means, yerr=errs, fmt="none", capsize=4, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(["Controls", "Patients"])
    ax.set_xlim(x[0] - 0.22, x[-1] + 0.22)
    ax.margins(x=0.01)
    ax.set_ylabel(ylabel, rotation=90, labelpad=14, va="center")
    ax.set_title(title, loc="center", pad=10)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close(fig)


def print_analysis(name, res):
    print(f"\n{name}")
    print(f"N control={res['n_control']}, N patient={res['n_patient']}")
    for ef in ["group", "condition", "interaction"]:
        e = res[ef]
        print(f"  {ef}: F({e['df1']}, {e['df2']:.2f})={e['F']:.3f}, p={e['p']:.4g}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    overall = summarize_overall()
    pd.Series(overall).to_csv(f"{OUT_DIR}/overall_summary.csv", header=["value"])

    analyses = {}

    # Gather wide-form dataframes from the windowed metrics and phase metrics
    window_wides = window_metric_dfs()
    ph_start = phase_metric_df("start")
    ph_end = phase_metric_df("end")

    # Helper to get subject sets by group from a wide dataframe
    def subjects_by_group(df):
        return {
            "control": set(df[df["group"] == "control"]["subject"].tolist()),
            "patient": set(df[df["group"] == "patient"]["subject"].tolist()),
        }

    # Non-phase subjects (union across all non-phase metrics)
    nonphase_ctrl = set()
    nonphase_pat = set()
    for wide in window_wides.values():
        s = subjects_by_group(wide)
        nonphase_ctrl |= s["control"]
        nonphase_pat |= s["patient"]

    # Phase subjects (union across start/end)
    phase_ctrl = subjects_by_group(ph_start)["control"] | subjects_by_group(ph_end)["control"]
    phase_pat = subjects_by_group(ph_start)["patient"] | subjects_by_group(ph_end)["patient"]

    # Intersection: only keep subjects present in both pipelines
    common_ctrl = nonphase_ctrl & phase_ctrl
    common_pat = nonphase_pat & phase_pat

    def filter_common(df):
        return df[
            (
                (df["group"] == "control") & df["subject"].isin(sorted(common_ctrl))
            )
            | (
                (df["group"] == "patient") & df["subject"].isin(sorted(common_pat))
            )
        ].reset_index(drop=True)

    # Apply filtering and run the split-plot ANOVA equivalent on the filtered sets
    for k, wide in window_wides.items():
        filtered = filter_common(wide)
        analyses[k] = split_plot_anova_equivalent(filtered)
        analyses[k]["wide"] = filtered

    ph_start_f = filter_common(ph_start)
    ph_end_f = filter_common(ph_end)
    analyses["phase_locking_saccade_onset"] = split_plot_anova_equivalent(ph_start_f)
    analyses["phase_locking_saccade_onset"]["wide"] = ph_start_f
    analyses["phase_locking_fixation_onset"] = split_plot_anova_equivalent(ph_end_f)
    analyses["phase_locking_fixation_onset"]["wide"] = ph_end_f

    print("OVERALL")
    for k, v in overall.items():
        print(f"{k}: {v}")

    summary_rows = []
    plot_info = {
        "saccade_frequency": ("Saccade Frequency", "Frequency (Hz)", "saccade_frequency_bar.png"),
        "saccade_duration_ms": ("Saccade Duration", "Duration (ms)", "saccade_duration_bar.png"),
        "mean_saccade_speed": ("Mean Saccade Speed", "Speed (deg/s)", "mean_saccade_speed_bar.png"),
        "max_saccade_speed": ("Maximum Saccade Speed", "Speed (deg/s)", "max_saccade_speed_bar.png"),
        "fixation_frequency": ("Fixation Frequency", "Frequency (Hz)", "fixation_frequency_bar.png"),
        "fixation_duration_ms": ("Fixation Duration", "Duration (ms)", "fixation_duration_bar.png"),
        "phase_locking_saccade_onset": ("Phase Locking: Saccade Onset", "Resultant Vector Length (r)", "phase_saccade_onset_bar.png"),
        "phase_locking_fixation_onset": ("Phase Locking: Fixation Onset", "Resultant Vector Length (r)", "phase_fixation_onset_bar.png"),
    }

    for name, res in analyses.items():
        print_analysis(name, res)
        res["cell_stats"].to_csv(f"{OUT_DIR}/{name}_cell_stats.csv", index=False)
        res["wide"].to_csv(f"{OUT_DIR}/{name}_subject_wide.csv", index=False)

        title, ylabel, fname = plot_info[name]
        plot_cells(res["cell_stats"], title, ylabel, f"{OUT_DIR}/{fname}")

        for ef in ["group", "condition", "interaction"]:
            e = res[ef]
            summary_rows.append(
                {
                    "analysis": name,
                    "effect": ef,
                    "F": e["F"],
                    "df1": e["df1"],
                    "df2": e["df2"],
                    "p": e["p"],
                    "n_control": res["n_control"],
                    "n_patient": res["n_patient"],
                }
            )

    pd.DataFrame(summary_rows).to_csv(f"{OUT_DIR}/anova_summary.csv", index=False)


if __name__ == "__main__":
    main()
