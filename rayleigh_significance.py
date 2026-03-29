import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel
from statsmodels.stats.multitest import multipletests
from pathlib import Path

from config import get_common_participant_ids
from rayleigh import phase_dist_window, rayleigh_test, rayleigh_window_group_df

WINDOWS = ((1, 4), (4, 7))
WHICH = "end"  # "start" = saccade onset, "end" = fixation onset
ALPHA = 0.05
N_SHUFFLES = 100
RNG_SEED = 20260326


def jaccard_index(a, b):
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def participant_rayleigh_rows(which=WHICH, windows=WINDOWS, alpha=ALPHA):
    rows = []
    summaries = []

    for group in ("ctrl", "pat"):
        ids = get_common_participant_ids(group, windows=windows)
        for hemi in (0, 1):
            sig_count = 0
            tested_count = 0

            for pid in ids:
                phases_by_window = []
                for window in windows:
                    phases = phase_dist_window(
                        pid,
                        group=group,
                        which=which,
                        window=window,
                        final=False,
                        hemi=hemi,
                    )
                    phases = np.asarray(phases, dtype=float)
                    phases = phases[np.isfinite(phases)]
                    if phases.size:
                        phases_by_window.append(phases)

                if not phases_by_window:
                    continue

                phases_all = np.concatenate(phases_by_window)
                z, p = rayleigh_test(phases_all)
                r = np.abs(np.mean(np.exp(1j * phases_all)))
                is_sig = p < alpha
                tested_count += 1
                sig_count += int(is_sig)

                rows.append(
                    {
                        "group": group,
                        "hemi": int(hemi),
                        "participant": int(pid),
                        "n": int(phases_all.size),
                        "r": float(r),
                        "z": float(z),
                        "p": float(p),
                        "sig_p_0p05": bool(is_sig),
                    }
                )

            if tested_count > 0:
                binom_res = binomtest(sig_count, tested_count, p=alpha, alternative="greater")
                summaries.append(
                    {
                        "group": group,
                        "hemi": int(hemi),
                        "k_sig": int(sig_count),
                        "n_participants": int(tested_count),
                        "p_binom": float(binom_res.pvalue),
                    }
                )

    if summaries:
        pvals = [x["p_binom"] for x in summaries]
        rej, p_fdr, _, _ = multipletests(pvals, method="fdr_bh")
        for i, x in enumerate(summaries):
            x["p_binom_fdr"] = float(p_fdr[i])
            x["sig_binom_fdr"] = bool(rej[i])

    return pd.DataFrame(rows), pd.DataFrame(summaries)


def jaccard_overlap_shuffle_p(participant_df, n_shuffles=N_SHUFFLES, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    rows = []

    for group in ("ctrl", "pat"):
        g = participant_df[participant_df["group"] == group]
        pivot = g.pivot_table(index="participant", columns="hemi", values="sig_p_0p05", aggfunc="first")
        if 0 not in pivot.columns or 1 not in pivot.columns:
            continue

        pivot = pivot.dropna(subset=[0, 1])
        participants = pivot.index.to_numpy(dtype=int)
        if participants.size == 0:
            continue

        left_sig = set(pivot.index[pivot[0].astype(bool)])
        right_sig = set(pivot.index[pivot[1].astype(bool)])
        observed = jaccard_index(left_sig, right_sig)

        null_vals = []
        left_n = len(left_sig)
        right_n = len(right_sig)
        for _ in range(int(n_shuffles)):
            left_perm = set(rng.choice(participants, size=left_n, replace=False))
            right_perm = set(rng.choice(participants, size=right_n, replace=False))
            null_vals.append(jaccard_index(left_perm, right_perm))

        null_vals = np.asarray(null_vals, dtype=float)
        p_perm_ge = float((1 + np.sum(null_vals >= observed)) / (n_shuffles + 1))

        rows.append(
            {
                "group": group,
                "n_with_both_hemi": int(participants.size),
                "k_sig_left": int(left_n),
                "k_sig_right": int(right_n),
                "k_sig_both": int(len(left_sig & right_sig)),
                "k_sig_union": int(len(left_sig | right_sig)),
                "jaccard_observed": float(observed),
                "jaccard_null_mean": float(np.nanmean(null_vals)),
                "jaccard_null_sd": float(np.nanstd(null_vals, ddof=1) if len(null_vals) > 1 else np.nan),
                "p_perm_ge_observed": p_perm_ge,
                "n_shuffles": int(n_shuffles),
            }
        )

    return pd.DataFrame(rows)


def pooled_window_ttests(which=WHICH, windows=WINDOWS):
    out = []
    for hemi in (0, 1):
        df_ctrl = rayleigh_window_group_df(group="ctrl", hemi=hemi, which=which, windows=windows, final=False)
        df_pat = rayleigh_window_group_df(group="pat", hemi=hemi, which=which, windows=windows, final=False)
        df = pd.concat([df_ctrl, df_pat], ignore_index=True)
        if df.empty:
            continue

        pivot = (
            df.assign(subject_uid=df["group"].astype(str) + "_" + df["participant"].astype(str))
            .pivot_table(
                index="subject_uid",
                columns=["window_start_s", "window_end_s"],
                values="r",
                aggfunc="mean",
            )
        )
        if windows[0] not in pivot.columns or windows[1] not in pivot.columns:
            continue

        a = pivot[windows[0]].to_numpy(dtype=float)
        b = pivot[windows[1]].to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            continue

        test = ttest_rel(a[mask], b[mask], nan_policy="omit")
        out.append(
            {
                "which": which,
                "hemi": int(hemi),
                "window_a": f"{windows[0][0]}-{windows[0][1]}",
                "window_b": f"{windows[1][0]}-{windows[1][1]}",
                "n": int(mask.sum()),
                "mean_r_a": float(np.nanmean(a[mask])),
                "mean_r_b": float(np.nanmean(b[mask])),
                "t": float(test.statistic),
                "p": float(test.pvalue),
            }
        )
    return pd.DataFrame(out)


def main():
    participant_df, binom_df = participant_rayleigh_rows()
    jaccard_df = jaccard_overlap_shuffle_p(participant_df)
    pooled_t_df = pooled_window_ttests()

    out_dir = Path("analysis_outputs/circular")
    out_dir.mkdir(parents=True, exist_ok=True)
    participant_df.to_csv(out_dir / f"rayleigh_participant_collapsed_{WHICH}.csv", index=False)
    binom_df.to_csv(out_dir / f"rayleigh_binomial_summary_{WHICH}.csv", index=False)
    jaccard_df.to_csv(out_dir / f"rayleigh_jaccard_overlap_{WHICH}.csv", index=False)
    pooled_t_df.to_csv(out_dir / f"rayleigh_pooled_window_ttests_{WHICH}.csv", index=False)

    print("Participant-level Rayleigh results (collapsed across both windows):")
    print(participant_df.to_string(index=False) if not participant_df.empty else "No rows")
    print("\nGroup-level binomial summaries:")
    print(binom_df.to_string(index=False) if not binom_df.empty else "No rows")
    print("\nLeft/right hemisphere overlap (Jaccard + shuffle p-value):")
    print(jaccard_df.to_string(index=False) if not jaccard_df.empty else "No rows")
    print("\nPooled paired t-tests of resultant length (fixation vs cue):")
    print(pooled_t_df.to_string(index=False) if not pooled_t_df.empty else "No rows")


if __name__ == "__main__":
    main()
