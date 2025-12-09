import os
from saccade_func import participant
from pathlib import Path

dir_path = Path(__file__).parent.resolve() / 'EyeData/controls'
n = len(os.listdir(dir_path))

for p in range(1, n + 1):
    print(f"\n=== Participant {p} ===")
    participant(p, show_stats=False, final=False)
