# GCN-former-TKAN

GCN-former-TKAN is a PyTorch workflow for **training and evaluating a time-series model that predicts the lateral displacement of deep-excavation retaining piles**. It combines spatial relations between monitoring points (GCN), temporal feature encoding (Transformer), and nonlinear sequence prediction (TKAN).

> This repository contains code only. It does not include project monitoring data, trained model weights, prediction tables, or figures.

## Who this is for

- Geotechnical and structural engineers monitoring deep excavations.
- Researchers studying data-driven prediction of retaining-structure deformation.
- Data scientists who have time-indexed excavation-monitoring data and want a reproducible PyTorch baseline.

It is not a real-time warning system by itself and must not replace site inspections, design checks, or engineering judgment.

## What the code does

For each prediction time, the model uses the preceding `seq_length` observations (default: 7) from three groups of monitoring variables:

| Group | Required columns | Role in the model |
| --- | --- | --- |
| `JG` | `JG1`–`JG11` | Engineering / construction-condition features |
| `AQ` | `AQ1`–`AQ17` | Auxiliary monitoring features |
| `J` | `J1`–`J20` | Retaining-pile displacement monitoring points |

The workflow chronologically splits the dataset into training and test periods, fits normalization only on the training portion, trains the model, and produces a next-step prediction for the configured target point (default: `J19`). During training, it saves the best model; after evaluation, it creates prediction tables, error metrics, and figures locally.

If a local adjacency matrix is absent, `main.py` calculates a 20×20 Spearman-correlation adjacency matrix from `J1`–`J20`. This means no precomputed matrix needs to be uploaded or stored in the repository.

## Repository contents

- `main.py`: main training and evaluation entry point.
- `models.py`: GCN, Transformer, and TKAN model definitions.
- `data_processor.py`: chronological splitting, normalization, and sequence construction.
- `evaluator.py`: inverse normalization, metrics, prediction tables, and figures.
- `spearman_utils.py`: Spearman-correlation adjacency-matrix generation.
- `experiment.py`: baseline and ablation experiment runner.
- `config.py`: data path, target point, model, and training settings.

## Before you run it

1. Install Python 3.10 or later. The code was verified with Python 3.10.15 and PyTorch 2.4.1.
2. Download or clone this repository.
3. Create a local `data` directory inside `GCN-former-TKAN`.
4. Put your private Excel workbook in that directory as `preprocessed_data_no_normalization.xlsx`.

Your local layout should look like this:

```text
GCN-former-TKAN/
├── data/
│   └── preprocessed_data_no_normalization.xlsx  # private; never commit
├── config.py
├── main.py
├── models.py
└── requirements.txt
```

## Input workbook specification

The Excel workbook must contain one row per monitoring time. It must include:

- An `index` column that can be converted to a timestamp.
- All numeric feature columns `JG1`–`JG11`, `AQ1`–`AQ17`, and `J1`–`J20`.
- No missing values in the columns used for training.
- Rows ordered chronologically, from oldest to newest.

The selected target point (for example, `J19`) must be included among `J1`–`J20`, because it is the value predicted at the next time step.

## Installation

From the `GCN-former-TKAN` directory, create and activate an environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a computer with NVIDIA CUDA support, install the PyTorch build appropriate to your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) before or instead of the generic `torch` entry in `requirements.txt`.

## Configure a run

Open `config.py` and adjust these fields in `DATA_CONFIG`:

```python
'file_path': 'data/preprocessed_data_no_normalization.xlsx',
'target_point': 'J19',
'test_size': 0.2,
'seq_length': 7,
```

- `file_path`: local path to the private workbook.
- `target_point`: one of `J1`–`J20`.
- `test_size`: proportion reserved for the latest test period. The split is chronological, not random.
- `seq_length`: number of preceding observations used for each prediction.

You may also change `epochs`, `learning_rate`, `patience`, and network widths in `TRAIN_CONFIG`, `GCN_CONFIG`, `TKAN_CONFIG`, and `TRANSFORMER_CONFIG`.

## Train and evaluate

Run:

```bash
python main.py
```

The code automatically chooses CUDA when PyTorch detects it; otherwise, it runs on CPU. To evaluate an existing local weight file without training again, place `best_model.pth` in the project directory and run:

```bash
python main.py --load_model
```

## Outputs

All generated outputs are stored locally and ignored by Git:

- `best_model.pth`: best model parameters found during the run.
- `outputs/prediction_results.xlsx`: test predictions.
- `outputs/prediction_results_train.xlsx`: training-period predictions.
- `outputs/prediction_results/test_predictions.xlsx`: an additional saved test-prediction table.
- `outputs/prediction_figures/`: prediction curve, scatter plot, loss curve, and metrics text file.

Metrics include MAE, MAPE, RMSE, and R². Interpret them only against a properly held-out, chronologically later test period; strong training performance alone does not demonstrate predictive validity.

## Privacy and safe publishing

`.gitignore` excludes data, CSV/Excel files, weight files, outputs, figures, and Python caches. Before every push, run `git status` and verify that only source code and documentation are staged. Do not upload monitoring data, coordinates, site drawings, personal information, or proprietary model weights unless you have explicit permission.

## Troubleshooting

- **`FileNotFoundError`**: confirm the workbook is at `data/preprocessed_data_no_normalization.xlsx`, or update `file_path` in `config.py`.
- **Missing column error**: add the full required set of `JG`, `AQ`, and `J` columns, exactly matching the names above.
- **CUDA error**: set `TRAIN_CONFIG['device'] = 'cpu'` in `config.py`, or install the PyTorch/CUDA combination that matches your system.
- **Too few samples**: reduce `seq_length` or collect more chronologically ordered observations; each split needs more than the sequence length.
- **Poor generalization**: use a later hold-out period, inspect sensor quality and missingness, and tune the sequence length and model settings. Do not select settings solely from the test set.

## Review note

The Python source files have passed syntax compilation, project-module import, random-tensor forward-pass, and Spearman adjacency-generation checks using Python 3.10.15 and PyTorch 2.4.1. `experiment.py` was also updated to use the current `TKAN_CONFIG` / `tkan_config` interface; unimplemented BiGRU and MLP comparison settings were removed.

