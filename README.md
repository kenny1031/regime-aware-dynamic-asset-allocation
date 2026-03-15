# Regime-Aware Dynamic Asset Allocation

A research-oriented multi-model framework for market regime detection and dynamic multi-asset allocation, combining unsupervised learning, supervised machine learning, deep learning, portfolio optimisation, walk-forward backtesting, and reinforcement learning.

## Overview

Financial markets are not stationary. Expected returns, volatility, and cross-asset correlations change over time, especially during stress episodes. This makes static portfolio construction fragile: a portfolio calibrated to one market environment may perform poorly when the regime changes.

This project studies dynamic asset allocation through the lens of **market regimes**. Instead of assuming a single return-generating process, it models markets as evolving through latent states such as **Neutral**, **Risk-On**, and **Risk-Off**, and uses those states to drive portfolio construction.

The repository is built as a **multi-model research pipeline** spanning the full workflow from raw financial data to adaptive allocation:

- reproducible financial time-series data engineering
- regime detection with **HMM, GMM, and KMeans**
- supervised regime classification with **XGBoost**
- sequence-based regime classification with **LSTM**
- regime-conditioned portfolio optimisation
- walk-forward out-of-sample backtesting
- reinforcement learning prototype allocation with **PPO**

## Motivation

Classical mean-variance optimisation assumes that expected returns and covariance structure are sufficiently stable to be estimated from historical data and reused in the future. In real markets, this assumption often breaks down.

In practice:

- expected returns differ across market environments
- volatility clusters during stress periods
- correlations among risky assets often rise in drawdowns
- diversification benefits can weaken when they are needed most

This project builds a **regime-aware allocation framework** designed to adapt to these state-dependent changes. The core idea is simple:

1. detect or classify the current market regime,
2. estimate regime-specific return/risk structure,
3. allocate differently across regimes.

## Asset Universe

The portfolio universe consists of nine benchmark asset classes observed at monthly frequency.

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

This growth/defensive split is used throughout the project for feature engineering, regime interpretation, and allocation constraints.

## Data Pipeline

The project uses a reproducible data pipeline with separate **raw**, **interim**, and **processed** layers.

### Raw data
The raw input consists of benchmark index-level data sourced from Bloomberg-style benchmark spreadsheets.

### Cleaning and transformation
The pipeline performs:

- date and ticker standardisation
- spreadsheet/vendor error token handling
- manual handling of known pre-inception placeholder history
- transformation from cleaned index levels to monthly return series

### Core processed outputs
Key processed datasets include:

- `asset_metadata.csv`
- `index_levels_clean.csv`
- `monthly_returns_wide.csv`
- `monthly_returns.csv`
- `summary_stats.csv`
- `missingness_report.csv`

These processed outputs serve as the foundation for regime modelling, portfolio statistics, optimisation, and backtesting.

## Feature Engineering for Regime Detection

A regime-sensitive feature set was engineered from the monthly return panel to capture market direction, relative performance, volatility, co-movement, momentum, drawdown, and stress.

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

These features were designed to provide a compact representation of changing market conditions while remaining interpretable from a financial perspective.

## Regime Models

The project compares multiple regime-learning approaches across unsupervised, supervised, deep learning, and reinforcement learning settings.

### Hidden Markov Model (HMM)
The primary regime model is a 3-state Gaussian HMM. It serves as the main regime engine because it:

- captures temporal persistence in latent states
- produces economically interpretable regimes
- outputs posterior probabilities that support soft regime switching

The three HMM states are interpreted as:

- **Neutral**
- **Risk-On**
- **Risk-Off**

### Gaussian Mixture Model (GMM)
A probabilistic clustering baseline in feature space. Unlike HMM, GMM does not model transition dynamics, so it is useful mainly as a static clustering comparator.

### KMeans
A simple hard clustering baseline. KMeans provides a lightweight benchmark for whether regime-like structure is visible in the engineered feature space even without probabilistic or temporal modelling.

### XGBoost Regime Classifier
A supervised tabular classifier trained to predict HMM-implied regime labels from the engineered feature set. It is primarily used to test whether regime structure can be learned discriminatively from snapshot features.

### LSTM Regime Classifier
A sequence-based deep learning model trained on rolling windows of regime features. It is used to test whether temporal context improves regime classification relative to static tabular models.

## Main Findings from Regime Detection

Among the unsupervised models, the **HMM produced the most economically meaningful regime structure**.

### HMM regime interpretation
The inferred states align well with financial intuition:

- **Neutral**: moderate positive growth returns and medium stress
- **Risk-On**: strongest growth returns and fewer negative-return assets
- **Risk-Off**: negative growth returns, worse tail outcomes, and broader market weakness

This makes the HMM the most suitable model for downstream allocation.

### Regime-dependent covariance structure
One of the most important findings is that return covariance structure changes materially across regimes. In Risk-Off periods:

- risky asset variances increase sharply
- correlations among growth assets rise
- diversification benefits weaken

This directly motivates the use of **regime-conditioned portfolio optimisation** rather than a single static allocation model.

## Supervised and Deep Learning Results

The project also studies whether HMM-implied regimes can be approximated using supervised machine learning and deep learning.

### XGBoost
XGBoost learns meaningful regime signals from engineered features and highlights financially interpretable regime drivers such as:

- growth drawdown
- equity drawdown
- worst asset return
- average cross-asset correlation

However, it is weaker at minority-regime discrimination, particularly for the less frequent non-neutral states.

### LSTM
The LSTM classifier improves on XGBoost by incorporating temporal structure through rolling feature sequences.

Current results indicate:

- **XGBoost**: Accuracy ≈ 0.73, Macro F1 ≈ 0.44
- **LSTM**: Accuracy ≈ 0.78, Macro F1 ≈ 0.55

This suggests that regime classification benefits from sequence modelling, consistent with the fact that market regimes evolve over time rather than appearing as independent monthly snapshots.

## Portfolio Optimisation

Using the HMM regime assignments, the project estimates regime-specific:

- expected return vectors
- covariance matrices
- correlation matrices

For each regime, a constrained quadratic-utility portfolio is solved:

$$
\max_{\mathbf{w}} \; \mathbf{w}^\top \boldsymbol\mu - \frac{\lambda}{2} \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}
$$

subject to:

- full investment
- long-only weights
- asset-level bounds
- growth/defensive allocation constraints

This produces different portfolio weights for:

- Neutral
- Risk-On
- Risk-Off

The resulting portfolios reflect the fact that both return opportunities and diversification structure change across market states.

## Backtesting

### In-sample regime-switching backtest
A preliminary in-sample backtest showed strong outperformance of regime-aware switching relative to static benchmarks. These results were treated as intentionally optimistic because they benefit from in-sample information and are therefore best viewed as proof-of-concept evidence.

### Walk-forward out-of-sample HMM backtest
A more realistic expanding-window walk-forward backtest was then implemented.

Two HMM-based switching variants were tested:

- **WF_HMM_Hard**: switch to the single most likely current regime portfolio
- **WF_HMM_Soft**: use posterior-probability-weighted averaging across regime portfolios

Current out-of-sample evidence suggests that:

- both walk-forward HMM strategies remain profitable
- soft switching is more stable than hard switching
- regime-aware allocation appears to improve risk-adjusted performance relative to a static balanced benchmark

## Reinforcement Learning Prototype

To go beyond model-based regime switching, the project also includes a reinforcement learning prototype for end-to-end dynamic allocation.

### Environment
A custom monthly allocation environment was built using:

- regime features
- HMM posterior probabilities
- current portfolio weights

At each step, the agent observes the current market state and outputs a new long-only fully-invested portfolio allocation.

### Action mapping
The raw action vector is transformed into valid portfolio weights using a softmax mapping, ensuring:

- non-negative weights
- full investment
- differentiable action-to-weight transformation

### Reward
The reward function is currently defined as:

- next-month portfolio return
- minus transaction costs
- minus an optional risk penalty

### PPO baseline
A PPO agent was trained as an initial RL baseline. The RL pipeline is fully functional and learns non-random adaptive allocation behaviour. However, the current PPO results should be interpreted as a **proof-of-concept prototype**, not as a final out-of-sample investment claim, since stricter train/test isolation and benchmark alignment are still future work.

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
│   ├── rl/
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

### Reinforcement learning
- `src/rl/env.py`
- `src/rl/smoke_test.py`
- `src/rl/train_ppo.py`

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

### 5. Run RL prototype
```bash
python3 -m src.rl.smoke_test
python3 -m src.rl.train_ppo
```

## Project Highlights

- Built a reproducible multi-asset data pipeline from raw benchmark index levels to cleaned monthly return panels
- Engineered a regime-sensitive feature set capturing return spreads, volatility, momentum, correlation, drawdown, and market stress
- Compared multiple regime-learning approaches across unsupervised, supervised, deep learning, and reinforcement learning settings
- Used a 3-state Gaussian HMM as the primary regime model for portfolio construction
- Estimated regime-conditioned expected returns and covariance matrices for dynamic allocation
- Implemented constrained regime-aware portfolio optimisation
- Ran both in-sample and expanding-window walk-forward backtests
- Built a custom reinforcement learning allocation environment and trained a PPO prototype agent

## Model Comparison

| Model | Family | Temporal Structure | Probabilistic Output | Main Role | Main Strength | Current Limitation |
|------|--------|--------------------|----------------------|-----------|---------------|--------------------|
| HMM | Unsupervised latent-state model | Yes | Yes | Primary regime model | Captures regime persistence and supports soft switching | Some numerical instability in expanding-window fits |
| GMM | Unsupervised probabilistic clustering | No | Yes | Static clustering baseline | Flexible probabilistic clustering in feature space | No transition dynamics |
| KMeans | Unsupervised hard clustering | No | No (one-hot assignment only) | Simplest clustering baseline | Easy to interpret and fast to run | No persistence, more fragmented states |
| XGBoost | Supervised tabular ML | No | Yes | Pseudo-label regime classifier | Good feature importance and strong tabular baseline | Weak minority-regime recall |
| LSTM | Supervised deep learning | Yes | Yes | Sequence-based regime classifier | Uses temporal context and outperforms XGBoost on macro F1 | Some overfitting on small monthly sample |
| PPO | Reinforcement learning | Yes | Policy-based | Dynamic allocation prototype | Learns adaptive allocation directly from state and reward | Current results are proof-of-concept, not strict OOS evidence |

## Results Summary

### Regime Detection
Among the unsupervised models, the 3-state Gaussian HMM produced the most economically interpretable market states:

- **Neutral**: moderate positive growth returns and medium stress
- **Risk-On**: strongest growth returns and fewer negative-return assets
- **Risk-Off**: negative growth returns, larger drawdowns, and broader market weakness

Compared with GMM and KMeans, HMM better matched the temporal notion of market regimes because it models state persistence and transition dynamics.

### Supervised and Deep Learning Models
Using HMM-implied regime labels as pseudo-labels:

- **XGBoost** achieved approximately **0.73 accuracy** and **0.44 macro F1**
- **LSTM** achieved approximately **0.78 accuracy** and **0.55 macro F1**

This suggests that sequence-based modelling improves regime classification relative to static tabular classification, especially when regime dynamics have temporal dependence.

### Regime-Aware Portfolio Construction
Regime-conditioned return and covariance estimates showed that both first and second moments vary materially across market states. In Risk-Off periods:

- risky asset variances increase sharply
- correlations across growth assets rise
- diversification benefits weaken

This motivates regime-aware allocation instead of a single static mean-variance framework.

### Walk-Forward Backtesting
An expanding-window walk-forward HMM backtest showed that:

- both hard and soft regime-switching strategies remained profitable out of sample
- **soft switching** was more stable than hard switching
- regime-aware allocation appeared to improve risk-adjusted performance relative to a static balanced benchmark

### Reinforcement Learning Prototype
A PPO-based allocation prototype was trained in a custom monthly asset allocation environment. The RL pipeline is functional and learns non-random allocation behaviour, but the current PPO results should be interpreted as a proof of concept rather than a final out-of-sample investment claim.


## Limitations

This repository should be viewed as a research-oriented prototype rather than a production-grade investment system.

Current limitations include:

- HMM estimation can be numerically unstable in some expanding-window refits
- several benchmark series have shorter effective histories than others, which affects regime-specific estimation
- supervised regime classification remains challenging for minority states because the label distribution is imbalanced
- walk-forward benchmark alignment and comparison can still be tightened further
- regime labels are model-implied constructs rather than directly observable ground truth
- the current portfolio optimisation setup is intentionally simplified relative to a full institutional asset allocation process
- the reinforcement learning component is currently a proof-of-concept prototype rather than a fully isolated out-of-sample policy evaluation

## Future Work

Natural next steps for extending the project include:

- more rigorous walk-forward benchmark comparison on fully aligned evaluation windows
- transaction cost and turnover sensitivity analysis
- alternative HMM specifications, including different covariance assumptions and state counts
- class-balanced supervised learning approaches for minority-regime classification
- stronger sequence models with improved validation and regularisation
- stricter train/test separation for reinforcement learning experiments
- regime-aware policy learning with reinforcement learning under more realistic reward and risk constraints

## Author

**Kenny Yu**

This project was developed as a research-oriented multi-model framework for regime detection and dynamic asset allocation, with the goal of combining quantitative finance, machine learning, deep learning, and portfolio engineering in a single reproducible repository.