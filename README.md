# PatchTST vs iTransformer: Compute-Matched LTSF Reproduction

CS5782 Spring 2026 class project. We reproduce PatchTST (ICLR 2023)
and iTransformer (ICLR 2024) on three LTSF benchmarks (Weather,
Electricity, Traffic) under matched 20-epoch compute budget.

## TL;DR Finding

PatchTST achieves the lowest test MSE on **all 12 of 12** (dataset,
horizon) combinations tested. Even a simple linear baseline
(DLinear-I) beats iTransformer on 7/8 cells across Weather and
Electricity.

## Data

Datasets (Weather, Electricity, Traffic, ETT) are not bundled in this
repository due to size. Download from:

> https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

After downloading, place the CSV files where the scripts expect them
(see each script's `--csv_path` argument or default).

## Repository Structure

- `patchtst_custom/`        Custom PatchTST/64 implementation
- `itransformer_wrappers/`  Shell wrappers for thuml/iTransformer
- `patchtst_repo_scripts/`  Shell wrappers for yuqinie98/PatchTST (TBD: scp from server)
- `experiments/`            Standalone Python scripts (SCM, DLinear, figures)
- `notebooks/`              Jupyter notebooks for SCM ablation and DLinear
- `results_summary/`        Final figures used in the poster

## Quick Usage

### Custom PatchTST/64
```bash
cd patchtst_custom
bash run_patchtst64.sh /path/to/traffic.csv
```

### iTransformer (requires clone of thuml/iTransformer first)
```bash
git clone https://github.com/thuml/iTransformer.git
cp itransformer_wrappers/*.sh iTransformer/
cd iTransformer
bash run_weather_compare.sh
```

### SCM (Stochastic Channel Mixing) ablation
```bash
python experiments/run_scm_electricity.py
```

### DLinear-I at h=720
```bash
python experiments/run_dlinear_720.py
```

### Generate poster figures
```bash
python experiments/make_correlation_heatmap.py
python experiments/make_main_figure.py
python experiments/make_reproduction_figure.py
```

## References

- Nie et al. PatchTST. ICLR 2023.
- Liu et al. iTransformer. ICLR 2024.
- Zeng et al. DLinear. AAAI 2023.
