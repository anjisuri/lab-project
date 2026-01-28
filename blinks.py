import numpy as np
import matplotlib.pyplot as plt
from config import get_control_file

plt.rcParams["font.family"] = "Helvetica"

# plotting pupil dilation for a single trial (to identify eye blinks)
def pupil_dilation(participant, trial_number):
    fs = 200  # sampling frequency Hz
    samples = 1601

    # load npy file for participant
    file_path = get_control_file(participant)
    data = np.load(file_path)  # shape: (3, time, trials)

    pupil_data = data[2]  # pupil is channel 2
    trial_pupil = pupil_data[:, trial_number - 1]  # timepoints for this trial

    time = np.arange(samples) / fs

    plt.figure(figsize=(10,4))
    plt.plot(time, trial_pupil)
    plt.title(f'Pupil dilation: participant {participant} trial {trial_number}')
    plt.xlabel('Time (s)')
    plt.ylabel('Pupil dilation (au)')
    plt.tight_layout()
    plt.show()
