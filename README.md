# Regime-Aware Dynamic Asset Allocation

A multi-model research project on dynamic multi-asset portfolio allocation using market regime detection, supervised learning, deep learning, and regime-conditioned portfolio optimisation.

## Overview

This project studies how changing market regimes affect multi-asset portfolio construction. Instead of assuming a single stationary return distribution, it models financial markets as evolving through latent states such as **Neutral**, **Risk-On**, and **Risk-Off**. These regimes are detected using unsupervised learning methods and then used to construct regime-conditioned portfolios.

The project combines:

- **financial time-series data engineering**
- **regime detection with HMM, GMM, and KMeans**
- **supervised regime classification with XGBoost**
- **sequence-based regime classification with LSTM**
- **regime-aware portfolio optimisation**
- **walk-forward out-of-sample backtesting**

## Motivation

Traditional mean-variance optimisation assumes that expected returns, volatilities, and correlations are stable over time. In practice, financial markets are highly nonstationary:

- expected returns vary across market environments
- volatility clusters in stress periods
- cross-asset correlations often rise during drawdowns
- diversification benefits can shrink in crises

This project builds a regime-aware framework that adapts portfolio allocation to changing market states.

## Asset Universe

The portfolio universe contains nine monthly benchmark asset classes:

### Growth assets
- `AEQ` — Australian Listed Equity
- `ILE_H` — International Listed Equity Hedged
- `ILE_UH` — International Listed Equity Unhedged
- `ALP` — Australian Listed Property
- `ILP_H` — International Listed Property Hedged
- `ILI_H` — International Listed Infrastructure Hedged

### Defensive assets
- `AFI` — Australian Fixed Income
- `IFI_H` — International Fixed Income Hedged
- `CASH` — Cash

## Data Pipeline

The data pipeline is fully reproducible and separates raw, interim, and processed layers.

### Raw data
Raw benchmark index level data were obtained from Bloomberg-style benchmark worksheets.

### Cleaning and transformation
The pipeline:
- standardises dates and tickers
- replaces spreadsheet/vendor error tokens with missing values
- removes known placeholder pre-inception periods
- constructs monthly simple returns from cleaned index levels

### Processed outputs
Key processed datasets include:
- `asset_metadata.csv`
- `index_levels_clean.csv`
- `monthly_returns_wide.csv`
- `monthly_returns.csv`
- `summary_stats.csv`
- `missingness_report.csv`

## Feature Engineering for Regime Detection

A regime feature set was engineered from the monthly return panel to capture market direction, volatility, co-movement, and stress.

### Return and spread features
- `equity_proxy_ret`
- `growth_proxy_ret`
- `defensive_proxy_ret`
- `gd_spread`

### Volatility and dispersion features
- `AEQ_vol_12m`
- `ILE_UH_vol_12m`
- `growth_vol_12m`
- `cross_sec_vol`

### Momentum features
- `AEQ_mom_3m`
- `AEQ_mom_12m`
- `growth_mom_3m`
- `growth_mom_12m`

### Correlation features
- `corr_AEQ_AFI_12m`
- `corr_growth_def_12m`
- `avg_corr_all_12m`
- `avg_corr_growth_12m`

### Drawdown and stress features
- `AEQ_drawdown`
- `growth_drawdown`
- `worst_asset_ret`
- `num_negative_assets`

## Regime Models

The project compares multiple models for regime detection and classification.

### 1. Hidden Markov Model (HMM)
The primary regime model is a 3-state Gaussian HMM. It is used because it:
- captures temporal persistence
- produces economically interpretable latent states
- provides posterior probabilities for soft regime switching

The HMM states are interpreted as:
- **Neutral**
- **Risk-On**
- **Risk-Off**

### 2. Gaussian Mixture Model (GMM)
A probabilistic clustering baseline without transition dynamics.

### 3. KMeans
A simple hard clustering baseline on the engineered feature space.

### 4. XGBoost Regime Classifier
A supervised tabular model trained to predict HMM-implied regime labels from the engineered feature set.

### 5. LSTM Regime Classifier
A sequence-based deep learning classifier that uses rolling feature windows to predict HMM-implied regime labels.

## Main Findings from Regime Detection

The HMM produces the most economically meaningful regime structure.

### HMM regime interpretation
- **Neutral**: moderate positive growth returns, medium stress
- **Risk-On**: strongest growth returns, fewer negative-return assets
- **Risk-Off**: negative growth returns, worse tail outcomes, more broad market weakness

### Regime-dependent covariance structure
The estimated covariance and correlation matrices vary materially across regimes. In Risk-Off periods:
- risky asset variances increase sharply
- cross-asset correlations among growth assets rise
- diversification benefits weaken

This motivates regime-conditioned portfolio optimisation rather than static allocation.

## Supervised / Deep Learning Results

### XGBoost
XGBoost learns useful regime signals from engineered features and identifies regime drivers such as:
- growth drawdown
- equity drawdown
- worst asset return
- average cross-asset correlation

However, it is relatively weak on minority-state discrimination.

### LSTM
The LSTM classifier improves on XGBoost by using temporal context from rolling feature sequences.

Observed results:
- **XGBoost**: Accuracy ≈ 0.73, Macro F1 ≈ 0.44
- **LSTM**: Accuracy ≈ 0.78, Macro F1 ≈ 0.55

This suggests that regime classification benefits from sequence modelling rather than purely static tabular features.

## Portfolio Optimisation

Using the HMM regime labels, the project estimates regime-specific:
- expected return vectors
- covariance matrices
- correlation matrices

For each regime, a constrained quadratic utility portfolio is solved:

$$
\max_{\mathbf{w}} \; \mathbf{w}^\top \boldsymbol\mu - \frac{\lambda}{2} \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}
$$

subject to:
- full investment
- long-only weights
- asset-level bounds
- growth/defensive allocation constraints

This produces distinct regime-conditioned portfolios for:
- Neutral
- Risk-On
- Risk-Off

## Backtesting

### In-sample regime-switching backtest
A preliminary in-sample comparison showed that regime-aware switching strategies strongly outperformed static benchmarks, although these results were intentionally treated as optimistic due to look-ahead bias.

### Walk-forward out-of-sample HMM backtest
A more realistic expanding-window walk-forward backtest was implemented.

Two variants were tested:
- **WF_HMM_Hard**: switch to the most likely current regime portfolio
- **WF_HMM_Soft**: posterior-probability-weighted average of regime portfolios

Current out-of-sample results indicate that:
- both walk-forward HMM strategies remain profitable
- soft switching is more stable than hard switching
- regime-aware allocation appears to improve risk-adjusted performance versus a static balanced benchmark

## Repository Structure

```text
regime-aware-dynamic-asset-allocation/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── figures/
│   ├── eda/
│   └── regime/
├── notebooks/
├── src/
│   ├── backtest/
│   ├── data/
│   ├── portfolio/
│   ├── regime/
│   └── utils/
├── requirements.txt
└── README.md
```

## Key Scripts
### Data
* `src/data/load_data.py`
* `src/data/build_regime_features.py`

### Regime modelling
* `src/regime/detect/run_markov_detection.py`
* `src/regime/detect/run_kmeans_detection.py`
* `src/regime/detect/run_xgb_regime_classifier.py`
* `src/regime/detect/run_lstm_regime_classifier.py`

### Portfolio and backtesting
* `src/regime/portfolio/regime_statistics.py`
* `src/regime/portfolio/regime_optimisation.py`
* `src/regime/backtest/regime_switching_backtest.py`
* `src/regime/backtest/walkforward_hmm_backtest.py`

## How to Run
### 1. Install required modules
```bash
pip install -r requirements.txt
```

### 1. Build processed data
```bash
python3 -m src.data.load_data
python3 -m src.data.build_regime_features
```

### 2. Run regime models
```bash
python3 -m src.regime.detect.run_markov_detection
python3 -m src.regime.detect.run_kmeans_detection
python3 -m src.regime.detect.run_xgb_regime_classifier
python3 -m src.regime.detect.run_lstm_regime_classifier
```

### 3. Build regime-conditioned portfolio inputs
```bash
python3 -m src.portfolio.regime_statistics
python3 -m src.portfolio.regime_optimisation
```

### 4. Run backtests
```bash
python3 -m src.backtest.regime_switching_backtest
python3 -m src.backtest.walkforward_hmm_backtest
```

## Limitations
This project is a research prototype and has several limitations:
* HMM state estimation can be numerically unstable in some expanding-window fits
* some benchmark series have shorter histories than others
* class imbalance makes supervised regime classification harder for minority regimes
* current walk-forward benchmark align can still be refined
* regime labels are model-dependent and not directly observable ground truth
* current portfolio optimisation is still simpler than a full institutional asset allocation engine

## Future Work
Planned next steps include:
* RL for dynamic allocation
* stronger walk-forward benchmark comparisons
* transaction cost sensitivity analysis
* alternative HMM specifications
* class-balanced supervised classifiers
* more robust sequence models
* regime-aware policy learning with RL

## Author
Kenny Yu

This project was developed as a research-oriented multi-model framework for regime detection and dynamic asset allocation, with the goal of combining quantitative finance, machine learning, deep learning, and portfolio engineering in a single reproducible repository.