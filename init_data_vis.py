from scipy.io import loadmat
from pathlib import Path

base = ('/Users/anji/Desktop/lab project')
loadmat((base+'/eye_tracking_data/controls'))
loadmat((base+'/eye_tracking_data/patients'))