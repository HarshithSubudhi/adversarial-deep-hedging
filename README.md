# Adversarial Deep Hedging: Learning to Hedge without Price Process Modeling

An end-to-end quantitative finance simulator and deep hedging engine based on the research paper: *\"Adversarial Deep Hedging: Learning to Hedge without Price Process Modeling\"* (Hirano, Minami, & Imajo, 2023).

---

## 📌 Project Overview
This repository implements a data-driven framework for option hedging in incomplete markets with transaction costs and regime shifts. Instead of assuming a parametric price process (like Heston or jump-diffusion), we train a neural network hedger and a recurrent neural network generator in an alternating **min-max game**. The generator learns to simulate unfavorable price paths, forcing the hedger to learn a robust, friction-aware, and model-free hedging strategy.

### Key Features:
*   **Path Simulators**: Implemented 3 stock price path simulators: Geometric Brownian Motion (GBM), a stressed Markov regime-switching SDE model, and a path-dependent Adversarial Generator.
*   **Friction-Aware Engine**: Models transaction costs ($c = 10$ bps) and dynamic rebalancing processes to calculate terminal portfolio wealth.
*   **Deep Hedging Optimizers**: Optimizes hedging policies directly using Entropic Risk Measure (ERM) utility and Conditional Value-at-Risk (CVaR) tail loss parameters.
*   **Comparative Backtests**: Evaluates performance across 10,000+ Monte Carlo simulations comparing Black-Scholes, Standard Deep Hedging, and Adversarial Deep Hedging.

---

## 🧮 Mathematical Formulations

### 1. Terminal P&L with Frictions
For a short-option position $Z$ (European Call) expiring at $T$ across $N$ steps, the terminal net Profit & Loss is:
$$\text{PL}_T = -Z(S_T) + \sum_{i=0}^{N-1} \delta_{t_i} (S_{t_{i+1}} - S_{t_i}) - \sum_{i=0}^{N} c S_{t_i} |\delta_{t_i} - \delta_{t_{i-1}}|$$
where $\delta_{t_i}$ is the hedge ratio (shares held), $\delta_{t_{-1}} = \delta_{t_N} = 0$, and $c$ is the proportional transaction cost.

### 2. Min-Max Optimization Game
The training objective is formulated as:
$$\min_H \max_G u\left(\text{PL}_T\left(Z, G(R), H(G(R))\right)\right)$$
where:
*   The **Hedger ($H$)** minimizes the loss function (maximizes utility).
*   The **Generator ($G$)** generates price paths to maximize the loss function (minimizes utility).
*   Loss is computed using the **Entropic Risk Measure (ERM)** with risk preference $\lambda$:
    $$L_{\text{ERM}} = \frac{1}{\lambda} \log \mathbb{E}\left[\exp(-\lambda \cdot \text{PL}_T)\right]$$

---

## 📁 Repository Structure
```filepath
├── data/
│   └── 20260205_option_minute_prices_expiry.csv   # Nifty index options minute price data
├── src/
│   ├── models.py          # PyTorch classes (DeepHedger, AdversarialGenerator)
│   ├── simulation.py      # Simulation routines (GBM, Stressed, Adversarial)
│   ├── engine.py          # DeltaHedgingEngine, risk metrics, and PyTorch P&L
│   ├── train.py           # Standard & alternating min-max training loops
│   └── hedging_utils.py   # Option Greeks, implied volatility, and CVaR calculations
└── notebooks/
    ├── 01_bs_baseline.ipynb               # Benchmark Black-Scholes delta hedging path
    └── 02_adversarial_deep_hedging.ipynb  # Main deep hedging experiments & visualizations
```

---

## 📈 Backtest Performance & Results

Evaluating option hedging strategies on $10,000$ Monte Carlo paths ($N=20$, $c=10$ bps):

### 1. Standard Market (GBM Paths)
Under calm market conditions matching Black-Scholes assumptions:

| Hedging Strategy | Mean P&L | Std Dev P&L | CVaR (95%) | Downside Risk | Hedging Error (RMSE) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Black-Scholes (BS)** | -0.0824 | 0.0154 | 0.1186 | 0.0154 | 0.0838 |
| **Standard Deep Hedging** | -0.0605 | 0.0912 | 0.3201 | 0.0912 | 0.1095 |
| **Adversarial Deep Hedging** | -0.0775 | 0.0358 | 0.1603 | 0.0358 | 0.0854 |

### 2. Stressed Market (Regime-Switching SDE Paths)
Under market stress (Markov regime-switching volatility spikes and market selloffs):

| Hedging Strategy | Mean P&L | Std Dev P&L | CVaR (95%) | Downside Risk | Hedging Error (RMSE) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Black-Scholes (BS)** | -0.0917 | 0.0432 | 0.2198 | 0.0432 | 0.1014 |
| **Standard Deep Hedging** | -0.0950 | 0.1339 | 0.4781 | 0.1339 | 0.1642 |
| **Adversarial Deep Hedging** | -0.0917 | 0.0549 | 0.2194 | 0.0549 | 0.1070 |

### 🔍 Key Quant Insights:
1.  **Regime Switch Robustness**: Standard deep hedging (trained only on normal GBM paths) completely breaks down under market stress, causing its CVaR to explode to **0.4781**. The **Adversarial Deep Hedger** maintains a tight CVaR of **0.2194**, proving highly robust.
2.  **No-Transaction Band**: In the presence of transaction costs, the neural network models learn a smoother delta profile relative to Black-Scholes (rebalancing less frequently on small price fluctuations) to optimize net P&L.

---

## 🚀 Setup & Execution

### Prerequisites
Install the required libraries:
```bash
pip install torch pandas numpy matplotlib jupyter
```

### Running the Project
1.  Navigate to the repository folder:
    ```bash
    cd quant-deep-hedging
    ```
2.  Launch Jupyter Lab/Notebook:
    ```bash
    jupyter notebook
    ```
3.  Open `notebooks/02_adversarial_deep_hedging.ipynb` to view the comprehensive deep hedging simulation.
