"""
Configuration file for lab-project
Handles file paths to work across different devices
"""
from pathlib import Path

# Get the project root directory (where this config.py file is located)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Define data directories
EYE_DIR = PROJECT_ROOT / 'EyeData'
CONTROLS_DIR = EYE_DIR / 'controls'
PATIENTS_DIR = EYE_DIR / 'patients'

# MEG analysis directory
MEG_DIR = PROJECT_ROOT / 'meg_data'
MEG_CTRLS = MEG_DIR / 'controls'
MEG_PAT = MEG_DIR / 'patients'

COMMON_CTRL_IDS = [1, 2, 3, 4, 8, 10, 14, 15, 17, 20, 21, 23, 24, 26]
COMMON_PAT_IDS = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 16, 17]

def list_control_ids(exclude=True):
    del exclude  # backward-compatible signature
    return list(COMMON_CTRL_IDS)

def list_patient_ids(exclude=True):
    del exclude  # backward-compatible signature
    return list(COMMON_PAT_IDS)

def get_control_file(participant):
    """Get path to control participant data file"""
    return CONTROLS_DIR / f'ctrl_{participant}.npy'

def get_patient_file(participant):
    """Get path to patient data file"""
    return PATIENTS_DIR / f'pat_{participant}.npy'

def get_meg_ctrl(participant):
    """Get path to control participant data file"""
    return MEG_CTRLS / f'meg_ctrl_{participant}.npy'

def get_meg_pat(participant):
    """Get path to patient data file"""
    return MEG_PAT / f'meg_pat_{participant}.npy'


def get_common_participant_ids(group, windows=((1, 4), (4, 7))):
    del windows  # kept for backward-compatible signature
    if group == "ctrl":
        return list(COMMON_CTRL_IDS)
    if group in ("pat", "patient"):
        return list(COMMON_PAT_IDS)
    return []


def get_common_participant_ids_by_group(windows=((1, 4), (4, 7))):
    del windows  # kept for backward-compatible signature
    return {
        "ctrl": list(COMMON_CTRL_IDS),
        "pat": list(COMMON_PAT_IDS),
    }
