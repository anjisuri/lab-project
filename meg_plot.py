import numpy as np
import matplotlib.pyplot as plt
from config import get_meg_ctrl, get_meg_pat

participant = 9
group = "ctrl"  # or pat
trial_count = 5

file_path = get_meg_ctrl(participant) if group == "ctrl" else get_meg_pat(participant)
data = np.load(file_path)  # shape: (trials, time, hemisphere)
print(data.shape)

left = data[:trial_count, :, 0]
right = data[:trial_count, :, 1]

fs = 200  # Hz
t = np.arange(left.shape[1]) / fs

fig, axes = plt.subplots(trial_count, 2, sharex=True, figsize=(15, 1.5 * trial_count))
for i in range(trial_count):
    axes[i, 0].plot(t, left[i])
    axes[i, 0].set_title(f"{group} p{participant} trial {i + 1} left")

    axes[i, 1].plot(t, right[i])
    axes[i, 1].set_title(f"{group} p{participant} trial {i + 1} right")

axes[-1, 0].set_xlabel("time (s)")
axes[-1, 1].set_xlabel("time (s)")

plt.tight_layout()
plt.show()
