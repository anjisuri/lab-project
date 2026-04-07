# Lab Project

Analysis pipeline for eye-tracking and MEG phase data in control vs patient groups.

The repository contains script-based analyses (plus notebooks) for:
- saccade and fixation metrics from eye-tracking trials,
- phase extraction from filtered MEG signals,
- circular/Rayleigh tests on phase locking,
- table exports for downstream ANOVA/statistical reporting.

## Repository Layout

- `config.py`: central paths and participant ID lists used across scripts.
- `sf_con.py`, `sf_pat.py`: trial- and participant-level saccade/fixation feature extraction.
- `mean_results.py`: group-level summary metrics and windowed metrics.
- `stats.py`: t-tests and ANOVA helper.
- `meg_filtering.py`: theta-band filtering + Hilbert transform for MEG signals.
- `rayleigh.py`: resultant vector length, Rayleigh tests, and window/group comparisons.
- `make_anova_tables.py`: exports ANOVA-ready CSVs to `analysis_outputs/anova_inputs/`.
- `mean_sem_metrics.py`: mean ± SEM summaries from ANOVA input tables.
- `circular_tests.py`: circular mean-difference tests between fixation and cue windows.
- `rayleigh_significance.py`: participant-level Rayleigh significance and overlap summaries.
- `performance_phase_tests.py`: links phase significance to performance metrics.
- `participant_vis.py`, `ctrl_data_vis.py`, `blinks.py`, `saccade_phase.py`: exploratory/diagnostic visualizations.
- `blinks.ipynb`, `results_figs.ipynb`: notebook analyses/figures.

## Data Expectations

By default, scripts expect this folder structure in the repo root:

```text
EyeData/
  controls/   # ctrl_<id>.npy
  patients/   # pat_<id>.npy
meg_data/
  controls/   # meg_ctrl_<id>.npy
  patients/   # meg_pat_<id>.npy
```

Expected array shapes used by the code:
- Eye data: `(3, time, trials)` with channels `[x, y, pupil]`.
- MEG data: `(trials, time, hemisphere)` where hemisphere index is `0=left`, `1=right`.

Participant IDs included by default are set in `config.py` (`COMMON_CTRL_IDS`, `COMMON_PAT_IDS`).

## Environment Setup

1. Create and activate a Python environment (Python 3.12 recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-image`, `statsmodels`, `mne`, `pycircstat2`.

## Typical Analysis Workflow

From the repository root:

1. Generate ANOVA-ready tables:

```bash
python make_anova_tables.py
```

2. Compute circular phase-difference tests:

```bash
python circular_tests.py --which start --hemi both
python circular_tests.py --which end --hemi both
```

3. Run collapsed Rayleigh significance summaries:

```bash
python rayleigh_significance.py
```

4. Print mean ± SEM summary table values:

```bash
python mean_sem_metrics.py
```

5. (Optional) Run additional statistical helpers in `stats.py` from an interactive session.

## Output Files

Generated outputs are written under `analysis_outputs/`, mainly:
- `analysis_outputs/anova_inputs/`: wide-format metric tables with `fixation` and `cue` columns.
- `analysis_outputs/circular/`: circular test outputs and Rayleigh summaries.
- `analysis_outputs/performance_phase/`: performance-vs-phase significance tables.

## Notes

- Paths are local/path-based (no package install needed); run scripts from repo root.
- `performance_phase_tests.py` currently reads an external Excel file via an absolute path; update the `xlsx` path in that script before running on another machine.
- The `lab_proj/` directory is a local virtual environment snapshot, not required for running analyses if you create your own environment.

