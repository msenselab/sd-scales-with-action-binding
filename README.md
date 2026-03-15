# Serial Dependence Scales with Action-Binding Depth in Duration Perception

Open data and analysis code for:

> Wu, J., Gündogdu, A., Akgün, H. C., Songur, C., & Shi, Z. (2025). Serial dependence scales with action-binding depth in duration perception.  [DOI to be added]

## Overview

This study tested whether serial dependence in time perception scales with the depth of action-binding. Participants (N = 24) performed duration reproduction and passive following tasks on identical stimuli. Sequential dependence was strongest when reproduction followed reproduction, intermediate when following preceded reproduction, and minimal when the current trial required following. Offset latency analysis revealed implicit duration encoding even during passive following.

## Directory Structure

```
public/
├── README.md
├── data/
│   └── dataAll.csv          # Trial-level data, all 24 participants
├── code/
│   ├── analysis.ipynb        # Main analysis notebook (reproduces all figures and statistics)
│   ├── analysis_utils.py     # Data processing and statistical utilities
│   ├── figure_utils.py       # Plotting functions
│   └── apa_utils.py          # APA-formatted statistical reporting
└── figures/
    ├── illustration.pdf      # Figure 1: Experimental paradigm
    ├── combined_cti_latency.pdf   # Figure 2: Central tendency + onset/offset latency
    ├── sequential_effects.pdf     # Figure 3: Sequential dependence across task transitions
    └── *.png                 # PNG versions of all figures
```

## Data Dictionary (dataAll.csv)

| Column | Description |
|--------|-------------|
| `nPar` | Participant ID (0–23) |
| `nB` | Block number (1–10) |
| `nT` | Trial number within block (1–56) |
| `curTask` | Current task: `reproduction` or `following` |
| `curDur` | Target duration on current trial (0.8–1.4 s) |
| `rpr` | Reproduced duration (0 if following trial) |
| `flw` | Followed duration (0 if reproduction trial) |
| `curPreCueDur` | Following stimulus duration (randomly chosen; NaN if reproduction) |
| `curRpt` | Current response duration |
| `preTask` | Previous trial task type (NaN for first trial in block) |
| `preDur` | Previous trial target duration |
| `preRpt` | Previous trial response duration |
| `prePreCueDur` | Previous trial following stimulus duration |
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
matplotlib
seaborn
scipy
statsmodels
pingouin
```

### Running

```bash
cd code/
jupyter notebook analysis.ipynb
# Run all cells — figures are saved to ../figures/
```

The notebook reproduces all statistics and figures reported in the manuscript.

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the material with appropriate attribution.

## Contact

Zhuanghua Shi — strongway@gmail.com
Department of Psychology, LMU Munich
