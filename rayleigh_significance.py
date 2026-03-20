import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from config import get_common_participant_ids
from rayleigh import phase_dist_window, rayleigh_test

windows = ((1, 4), (4, 7))
which = "end"  # "start" = saccade onset, "end" = fixation onset
alpha = 0.05

# collapse across both time windows per participant, then test each participant separately
# repeat for each hemisphere
participant_rows = []
summary_rows = []

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
                phases = np.asarray(phases)
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

            participant_rows.append(
                {
                    "group": group,
                    "hemi": hemi,
                    "participant": pid,
                    "n": int(phases_all.size),
                    "r": float(r),
                    "z": float(z),
                    "p": float(p),
                    "sig_p_0p05": bool(is_sig),
                }
            )

        if tested_count > 0:
            binom_res = binomtest(sig_count, tested_count, p=alpha, alternative="greater")
            summary_rows.append(
                {
                    "group": group,
                    "hemi": hemi,
                    "k_sig": int(sig_count),
                    "n_participants": int(tested_count),
                    "p_binom": float(binom_res.pvalue),
                }
            )

# FDR-correct binomial tests across all group x hemisphere summaries.
if summary_rows:
    pvals = [x["p_binom"] for x in summary_rows]
    rej, p_fdr, _, _ = multipletests(pvals, method="fdr_bh")
    for i, x in enumerate(summary_rows):
        x["p_binom_fdr"] = float(p_fdr[i])
        x["sig_binom_fdr"] = bool(rej[i])

print("Participant-level Rayleigh results (collapsed across both windows):")
for row in participant_rows:
    print(row)

print("\nGroup-level binomial summaries:")
for row in summary_rows:
    print(row)