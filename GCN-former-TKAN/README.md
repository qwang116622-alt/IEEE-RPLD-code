# GCN-former-TKAN

Code for predicting lateral displacement of retaining piles in deep excavations with a GCN, Transformer encoder, and TKAN prediction head.

## Repository contents

- `main.py`: training and evaluation entry point.
- `models.py`: GCN, Transformer, and TKAN model components.
- `data_processor.py`: chronological split, normalization, and sequence dataset construction.
- `evaluator.py`: metrics and result visualizations.
- `spearman_utils.py`: Spearman-correlation adjacency-matrix generation.
- `experiment.py`: baseline and ablation experiment runner.
- `config.py`: model, training, input, and output configuration.

## Data policy

No monitoring data, trained weights, prediction tables, figures, or Python cache files are included. Place your private input workbook at:

`data/preprocessed_data_no_normalization.xlsx`

The workbook must use `index` as its timestamp column and contain `JG1`–`JG11`, `AQ1`–`AQ17`, and `J1`–`J20` columns. These files are ignored by Git.

## Setup and run

```bash
python -m pip install -r requirements.txt
python main.py
```

The program writes weights and evaluation artifacts to `outputs/`, which is also ignored by Git. If no local precomputed adjacency matrix is present, `main.py` builds one from the local data using Spearman correlation.

## Review note

The training entry point has been syntax-checked. `experiment.py` was updated to use the current `TKAN_CONFIG` / `tkan_config` interface; its obsolete BiGRU and MLP comparisons were removed because those model variants are not implemented in `models.py`.

