# Channel Charting for Position and Orientation

This repository contains the training and evaluation code used in [1].
It implements neural network (NN) training to jointly estimate the user equipment (UE) position and orientation from measured 5G channel-state information (CSI).

The implementation supports two learning methods:

- **Channel charting:** self-supervised training with state-of-the-art
  triplet, bilateration, and bounding-box losses for position [2, 3], extended by a novel orientation triplet [1] and orientation alignment [1] loss.
- **Neural positioning:** supervised training using ground-truth position and
  orientation labels.

Each method can use either predefined hyperparameters or an Optuna
multi-objective parameter search.

## Code provenance

This codebase is based on the
[`caez-5g` branch of IIP-Group/ChannelCharting_RealWorld](https://github.com/IIP-Group/ChannelCharting_RealWorld/tree/caez-5g).
It extends and reorganizes that implementation for joint estimation of position and orientation (channel charting for position and orientation), provides a supervised baseline as a comparison (joint neural positioning with orientation), and offers fixed parameter and Optuna parameter optimization training modes.

The authors used large language models during preparation of this codebase. All outputs were independently evaluated and verified by the authors.

## Dataset

The experiments use the **CAEZ-5G-OUTDOOR** dataset. Dataset descriptions,
license terms, download links, and preprocessed CSI features are available at [https://caez.ethz.ch](https://caez.ethz.ch).

Before running the training program, download the dataset and preprocess the dataset files with the
[`gen_dataset.py`](gen_dataset.py) script. Configure its input data path
and feature-processing options in the script before running it. It generates
the `.npz` files described below for training. The script is based on
[`gen_dataset.py` from IIP-Group/neural-positioning](https://github.com/IIP-Group/neural-positioning/blob/main/gen_dataset.py)
and was modified to include UE orientation data in the generated dataset.

The training program expects these preprocessed files:

```text
robot_measurement_outdoor_2025_10_11_auto_correlation_no_norm_truncated_25_sorted_4_rx_ant_wo_last_10000.npz
robot_measurement_outdoor_2025_10_11_auto_correlation_no_norm_truncated_25_sorted_4_rx_ant_only_last_10000.npz
```

By default, `DATA_DIR` points to a `datasets` directory next to the project
directory:

```text
workspace/
├── datasets/
│   ├── robot_measurement_outdoor_2025_10_11_auto_correlation_no_norm_truncated_25_sorted_4_rx_ant_wo_last_10000.npz
│   └── robot_measurement_outdoor_2025_10_11_auto_correlation_no_norm_truncated_25_sorted_4_rx_ant_only_last_10000.npz
└── <project-directory>/
    └── train_5g_real_world_orientation.py
```

If you use another layout, change `DATA_DIR`, `PRIMARY_DATASET`, and
`HELD_OUT_DATASET` in `train_5g_real_world_orientation.py`.

The primary NPZ archive must contain `oru_pos_dict`, `H`, `pos`, `rot`,
`timestamps`, and `bounding_box`. The held-out archive must contain `H`,
`pos`, `rot`, and `timestamps`.

## Orientation dependence of CSI

The [`CAEZ_5G_Orientation_Exploration.ipynb`](CAEZ_5G_Orientation_Exploration.ipynb) notebook 
explores how UE rotation affects the measured 5G CSI. It is a supporting
raw data analysis and does not train either of the models described below. The
notebook:

- aligns channel estimate timestamps with motion capture positions and
  quaternions
- detects five rotation measurement intervals from yaw zero crossings
- plots the UE trajectory and normalized channel magnitude waterfalls
- produces channel plots showing the orientation dependence.

Unlike the training script, the notebook operates directly on the extracted,
unprocessed downloadable dataset. Set the `CAEZ_5G_DATA_DIR` environment variable
or edit `DATA_DIR` in the parameters cell before running it.

Open the notebook with the `cc_env` Python kernel and run its cells in
order. By default, its six plots are saved as PDFs in `figures/`. Set
`FIGURE_OUTPUT_DIR = None` in the parameters cell to display them without
writing files. The timestamp offset, motion capture timezone, loading worker
count, and batch size are configurable in the same cell.

## Python environment preparation with Anaconda

A CUDA-capable GPU is strongly recommended for training. The supplied
`environment.yml` contains a reference set of dependencies used for this project, including
Python 3.11, PyTorch, Optuna, the CUDA runtime libraries, and all transitive
dependencies. Other version of the dependencies might also work.

Create and activate the environment with Anaconda or Miniconda:

```bash
conda env create --name cc_env --file environment.yml
conda activate cc_env
```

For GPU execution, ensure that the installed PyTorch build is compatible with
the NVIDIA driver and CUDA runtime on your system. Verify the installation
with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If the final line is `False`, the program falls back to CPU execution.

## Configuration

Select the learning method and parameter mode near the top of
`train_5g_real_world_orientation.py`:

```python
METHOD = "channel_charting"  # Or "neural_positioning".
USE_OPTUNA = False           # Set to True to run or resume Optuna.
```

The two flags provide four configurations:

| Learning method | Parameter mode | Description |
| --- | --- | --- |
| Channel charting | Fixed | Train once with the predefined channel-charting configuration. |
| Channel charting | Optuna | Tune the channel-charting losses and optimizer parameters. |
| Neural positioning | Fixed | Train once with the supervised configuration. |
| Neural positioning | Optuna | Tune supervised optimizer and orientation-loss parameters. |

Other commonly changed settings are:

```python
CUDA_DEVICE_INDEX = 0
OPTUNA_TRIALS = 100
SAVE_RESULTS = True
EVALUATE_TRAINING_SET = False
SAVE_OPTUNA_TRIAL_ARTIFACTS = True
```

Optuna mode stores its study in SQLite and resumes completed trials when the
same method specific study already exists. It selects the Pareto front trial
with the lowest validation mean distance error and retrains a final model.
The held-out test set is evaluated only after model selection.

## Running an experiment

From the project directory, run:

```bash
conda activate cc_env
python train_5g_real_world_orientation.py
```

The program prints the selected method, parameter mode, device, dataset paths,
and final validation and held-out metrics. KS, trustworthiness, continuity,
position error, and orientation error are evaluated on every sample in the
respective split.

## Outputs

Generated artifacts are written to method- and mode-specific directories:

```text
results/
├── channel_charting/
│   ├── fixed_hyperparameters/
│   └── optuna_search/
└── neural_positioning/
    ├── fixed_hyperparameters/
    └── optuna_search/
```

With `SAVE_RESULTS = True`, the final model run writes:

- `trained_model_state_dict.pth`: final trained model weights
- `training_configuration.pkl`: serialized final training configuration
- `evaluation_metrics.json`: method, mode, hyperparameters, and evaluation results
- `training_loss_curves.png`: training loss curve for the selected method, with an additional validation loss curve when neural positioning is selected
- `ground_truth_chart.png`: color coded ground-truth positions from the primary
  dataset (training and validation), with grayscale orientation arrows for a
  selected subset of held-out test samples
- `test_and_validation_chart.png`: validation split chart with grayscale
  orientation arrows marking a selected subset of held-out test samples
- `validation_orientation_predictions.csv` and `held_out_orientation_predictions.csv`:
  per-sample predicted and ground-truth `phi`, with timestamps

When `USE_OPTUNA = True`, the `optuna_search` directory also contains
`optuna_study.sqlite3`, `optuna_trials.csv`, and
`selected_optuna_trial.json`. With `SAVE_OPTUNA_TRIAL_ARTIFACTS = True`, each
trial run saves artifacts under `trial_artifacts/trial_NNN/`:
`training_loss_curves.png`, `test_and_validation_chart.png`, and
`trial_summary.json`.

## Repository structure

```text
.
├── train_5g_real_world_orientation.py       # Unified training and evaluation entry point
├── gen_dataset.py                      # Local preprocessing for raw CSI as well as position and orientation logs
├── CAEZ_5G_Orientation_Exploration.ipynb # Supporting dataset exploration
├── figures/                            # Supporting analysis figures
├── utils/                              # Models, losses, metrics, and data helpers
├── results/                            # Generated method/mode-specific outputs
├── environment.yml                     # Reproducible Anaconda environment
└── README.md
```

## Citation

If you use our code, or parts of it, you must cite the following reference:

```bibtex
@misc{richner2026channel,
  author        = {Richner, Daniel and Wiesmayr, Reinhard and Zumegen, Frederik and Studer, Christoph},
  title         = {Channel Charting for Position and Orientation},
  year          = {2026},
  month         = {Jun.},
  journal       = {arXiv:2606.18151}
}
```

If you use our CAEZ-5G datasets, or parts of it, you must cite the following reference:

```bibtex
@inproceedings{wiesmayr2025csi,
  author    = {Wiesmayr, Reinhard and Zumegen, Frederik and Taner, Sueda and Dick, Chris and Studer, Christoph},
  title     = {{CSI}-Based User Positioning, Channel Charting, and Device Classification with an {NVIDIA 5G} Testbed},
  booktitle = {Asilomar Conference on Signals, Systems, and Computers},
  month     = {Oct.},
  year      = {2025}
}
```

## References

[1] D. Richner, R. Wiesmayr, F. Zumegen, and C. Studer, "Channel Charting for Position and Orientation," [arXiv:2606.18151](https://arxiv.org/abs/2606.18151), Jun. 2026.

[2] P. Ferrand, A. Decurninge, L. G. Ordóñez, and M. Guillaud, “Triplet-
based wireless channel charting: Architecture and experiments,” IEEE J.
Sel. Areas Commun., vol. 39, no. 8, pp. 2361–2373, 2021.

[3] S. Taner, V. Palhares, and C. Studer, “Channel charting in real-world
coordinates with distributed MIMO,” IEEE Trans. Wireless Commun.,
vol. 24, no. 9, pp. 7286–7300, 2025.
