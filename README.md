# PatchTST vs iTransformer: Compute-Matched LTSF Reproduction

This repo contains a re-implementation of PatchTST (Nie et al., ICLR 2023), a patch-based Transformer for long-term time-series forecasting (LTSF). PatchTST's core contribution is channel independence (CI). We benchmark it head-to-head against iTransformer (channel-dependent, CD) and DLinear (parameter-efficient linear baseline) under identical training conditions, and introduce Stochastic Channel Mixing (SCM) to probe the CI/CD trade-off.

## TL;DR Finding

PatchTST achieves the lowest test MSE on **all 12 of 12** (dataset,
horizon) combinations tested. Even a simple linear baseline
(DLinear-I) beats iTransformer on 7/8 cells across Weather and
Electricity. As the correlation between the channels/features of the dataset increase, the performance difference between CI and CM reduces. Results suggest that a mix of both strategy would be beneficial to learn the optimal amount of channel characteristics. We provide this evidence using SCM.

## Data

Datasets (Weather, Electricity, Traffic, ETT) are not bundled in this
repository due to size. Download from:

> https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

After downloading, place the CSV files where the scripts expect them
(see each script's `--csv_path` argument or default).


## Re-implementation Details
**Models:** PatchTST (patch length 16, stride 8; variants /64 and /42), iTransformer (inverted token per variate), DLinear (moving-average decomposition + linear layers), and SCM (lightweight linear mixer interpolating CI/CD encodings via α ∈ {0, 0.25, 0.50, 0.75, 1.0}).

**Datasets:** Weather (21 variates), Electricity (321 variates), Traffic (862 variates) — same splits as the original paper.

**Evaluation:** MSE and MAE at horizons {96, 192, 336, 720}. All models share identical data loaders, reversible instance normalization, and evaluation routines.

**Key modifications vs. original:** Training capped at 20 epochs (vs. 100 in the paper) using gradient accumulation to maintain effective batch size. This raises absolute MSE by ~2–8% but preserves relative rankings, validating compute-matched comparison as a practical evaluation paradigm.

## Repository Structure

- `patchtst_custom/`        Custom PatchTST/64 implementation
- `itransformer_wrappers/`  Shell wrappers for thuml/iTransformer
- `patchtst_repo_scripts/`  Shell wrappers for yuqinie98/PatchTST (TBD: scp from server)
- `experiments/`            Standalone Python scripts (SCM, DLinear, figures)
- `notebooks/`              Jupyter notebooks for SCM ablation and DLinear
- `results_summary/`        Final figures used in the poster
- `report/`                 Final report
- `poster/`                 Project poster

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

- Y. Nie, N. H. A. Nguyen, P. Sinthong, and J. Kalagnanam. "A time series is worth 64 words: Long-term forecasting with Transformers." ICLR, 2023.

- Y. Liu et al. "iTransformer: Inverted Transformers are effective for time series forecasting." ICLR, 2024.

- A. Zeng, M. Chen, L. Zhang, and Q. Xu. "Are Transformers effective for time series forecasting?" AAAI, 37(9), 2023.

- H. Wu, J. Xu, J. Wang, and M. Long. "Autoformer: Decomposition Transformers with auto-correlation for long-term series forecasting." NeurIPS, 2021.
