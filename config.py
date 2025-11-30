"""
Configuration file for lab-project
Handles file paths to work across different devices
"""
import os
from pathlib import Path

# Get the project root directory (where this config.py file is located)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Define data directories
DATA_DIR = PROJECT_ROOT / 'EyeData'
CONTROLS_DIR = DATA_DIR / 'controls'
PATIENTS_DIR = DATA_DIR / 'patients'

# MEG analysis directory
MEG_DIR = PROJECT_ROOT / 'meg_analysis'

def get_control_file(participant):
    """Get path to control participant data file"""
    return CONTROLS_DIR / f'ctrl_{participant}.npy'

def get_patient_file(participant):
    """Get path to patient data file"""
    return PATIENTS_DIR / f'pat_{participant}.npy'
