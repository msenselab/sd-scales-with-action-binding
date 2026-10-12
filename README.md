# Action Binding Gates Serial Dependence in Duration Perception

Open data and analysis code for:

> Wu, J., Gündogdu, A., Akgün, H. C., Songur, C., & Shi, Z. (under review). Action binding gates serial dependence in duration perception. [DOI to be added]

The manuscript was originally submitted as *Serial Dependence Scales with Action-Binding Depth in Duration Perception*.

## Overview

Twenty-four participants reproduced or followed visual durations (0.8–1.4 s) in randomly interleaved trials, with the task cued after the stimulus. In reproduction, participants held a key for the perceived duration; in following, a second stimulus appeared and participants held the key while it was visible. Stimuli and motor output were matched, while the binding of duration into the action differed.

- Current reproduction responses were attracted to the previous duration equally after reproduction (RR) and after following (FR).
- Current following responses were attracted only after another following trial (FF), not after reproduction (RF).
- Within a previous following trial, the interval participants acted on (T2) carried the attraction; the merely viewed pre-cue interval (T1) contributed less, reliably so before a following trial.
- Following still showed central tendency, located in key-release (offset) timing, indicating implicit duration encoding.

Serial dependence is therefore gated by action binding: whether the previous interval was bound into a response, and whether the current response retrieves that binding.

## Directory Structure

```
├── README.md
├── data/
│   └── dataAll.csv                     # Trial-level data, all 24 participants
├── code/
│   ├── run_analyses.py                 # Reproduces all statistics, Figures 2–4 and Tables S1–S4
│   ├── analysis_utils.py               # Data processing utilities (used by both analyses)
│   ├── analysis.ipynb                  # Analyses of the originally submitted manuscript
│   ├── figure_utils.py                 # Plotting functions (notebook)
│   └── apa_utils.py                    # APA-formatted statistical reporting (notebook)
├── results/                            # Written by run_analyses.py
│   ├── results.md / results.json       # All reported statistics
│   └── *.csv                           # Participant-level indices
└── figures/
    ├── illustration.pdf/.png           # Figure 1: experimental paradigm
    ├── fig2_central_tendency.pdf/.png  # Figure 2: central tendency and following-task latencies
    ├── fig3_sequential_dependence.pdf/.png  # Figure 3: bias vs relative previous duration
    ├── fig4_sdi_by_transition.pdf/.png      # Figure 4: SDI by transition
    └── (other files)                   # Figures of the originally submitted manuscript (notebook)
```

| Manuscript item | Source |
|---|---|
| Figure 1 | `figures/illustration.*` |
| Figures 2–4 | `figures/fig2_*`, `fig3_*`, `fig4_*` (from `run_analyses.py`) |
| All statistics in the text; Supplementary Tables S1–S4 | `results/results.md` (from `run_analyses.py`) |

## Data Dictionary (dataAll.csv)

Each following trial contains two intervals: a pre-cue interval (T1) shown before the task cue, which participants viewed but did not act on, and the post-cue interval they followed (T2). The two were drawn independently.

| Column | Description |
|--------|-------------|
| `nPar` | Participant ID (0–23) |
| `nB` | Block number (1–10) |
| `nT` | Trial number within block (1–56) |
| `curTask` | Current task: `reproduction` or `following` |
| `curDur` | Target duration on the current trial (0.8–1.4 s): the duration to be reproduced in reproduction trials; the followed interval T2 in following trials |
| `rpr` | Reproduced duration (0 if following trial) |
| `flw` | Followed (key-held) duration (0 if reproduction trial) |
| `curPreCueDur` | Pre-cue interval T1 of a following trial (viewed, not acted on); NaN for reproduction trials |
| `curRpt` | Current response duration |
| `preTask` | Previous trial task type (NaN for first trial in block) |
| `preDur` | Previous trial target duration (reproduced target, or T2 of a previous following trial) |
| `preRpt` | Previous trial response duration |
| `prePreCueDur` | Pre-cue interval T1 of the previous trial, if it was a following trial |
| `flwOnLtc` | Following onset latency (key press − stimulus onset) |
| `flwOffLtc` | Following offset latency (key release − stimulus offset) |
| `flwAfterOff` | Whether key press started after stimulus offset |
| `valid` | Trial validity flag (1 = valid) |

## Reproducing the Analysis

### Requirements

```
Python >= 3.9
pandas
numpy
scipy
statsmodels
pingouin
matplotlib
seaborn
```

For example: `pip install pandas numpy scipy statsmodels pingouin matplotlib seaborn`.

### Running

```bash
python code/run_analyses.py
```

This writes all statistics to `results/results.md` (and `results.json`, plus participant-level CSV files) and Figures 2–4 to `figures/`. Sequential dependence is estimated per participant from the joint model *B*<sub>n</sub> ~ *T*<sub>n</sub> + *T*<sub>n−1</sub>; the simple-slope specification without *T*<sub>n</sub> is reported for comparison (Table S1).

`code/analysis.ipynb` reproduces the analyses and figures of the originally submitted manuscript and is kept for transparency.

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the material with appropriate attribution.
