"""
Configuration file for lab-project
Handles file paths to work across different devices
"""
import os
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