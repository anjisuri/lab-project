import numpy as np
from statsmodels.stats.multitest import multipletests
from config import get_common_participant_ids
from rayleigh import phase_dist_window, rayleigh_test

windows = ((1, 4), (4, 7))
which = "start"  # "start"=saccade onset, "end"=fixation onset
rows = []

for group in ("ctrl", "pat"):
    ids = get_common_participant_ids(group, windows=windows)
    for window in windows:
        all_phases = []
        for pid in ids:
            p = phase_dist_window(pid, group=group, which=which, window=window, final=False, hemi = 1)
            p = np.asarray(p)
            all_phases.extend(p[np.isfinite(p)])
        all_phases = np.asarray(all_phases)

        z, p = rayleigh_test(all_phases)
        r = np.abs(np.mean(np.exp(1j * all_phases)))
        rows.append({"group": group, "window": window, "n": len(all_phases), "r": r, "z": z, "p": p})

# correct across the 4 planned tests
pvals = [x["p"] for x in rows]
rej, p_fdr, _, _ = multipletests(pvals, method="fdr_bh")
for i, x in enumerate(rows):
    x["p_fdr"] = float(p_fdr[i])
    x["sig_fdr"] = bool(rej[i])

for x in rows:
    print(x)
