# Adversarial Deep Hedging

This repository implements the adversarial deep hedging framework described in [Hirano, Minami, and Imajo (2023)](https://arxiv.org/pdf/2307.13217). It provides a modular Python implementation and Jupyter notebooks to train and evaluate neural network hedging policies in incomplete markets under transaction costs and regime-switching volatility.

---

## Model Framework

### 1. Terminal P&L with Frictions
For a written European call option $Z$ expiring at $T$ across $N$ discrete steps, the terminal net Profit & Loss (P&L) is:

$$PL_T = -Z(S_T) + \sum_{i=0}^{N-1} \delta_{t_i} (S_{t_{i+1}} - S_{t_i}) - \sum_{i=0}^{N} c S_{t_i} |\delta_{t_i} - \delta_{t_{i-1}}|$$

where $S_{t_i}$ is the asset price, $\delta_{t_i}$ is the shares held (with $\delta_{t_{-1}} = \delta_{t_N} = 0$), and $c$ is the transaction cost rate (e.g., 10 bps).

### 2. Min-Max Game
We train a **Hedger** ($H$) and an **Adversarial Generator** ($G$) in a min-max game:

$$\min_H \max_G \mathbb{E}\left[-u(\text{PL}_T)\right]$$

where $u(x)$ is the Entropic Risk Measure (ERM) utility:

$$u(x) = -\frac{1}{\lambda} \log \mathbb{E}\left[\exp(-\lambda x)\right]$$

*   **Hedger**: Optimizes option delta hedge positions to maximize utility.
*   **Generator**: Autoregressively simulates price paths designed to maximize the hedger's replication loss.

---

## Repository Structure
*   **[src/](file:///c:/Users/harsh/All%20in%20one/Quant%20Resources/AZ%20Quant%20Project/quant-deep-hedging/src)**: Modular Python code files.
    *   `models.py`: PyTorch networks for standard and adversarial deep hedging.
    *   `simulation.py`: Price path simulators (GBM, Stressed SDE, Adversarial GRU).
    *   `engine.py`: Valuation engine, risk metrics, and PyTorch P&L backpropagation logic.
    *   `train.py`: Differentiable training loops for standard and adversarial training.
    *   `hedging_utils.py`: Black-Scholes pricing, implied volatility search, and CVaR calculations.
*   **[notebooks/](file:///c:/Users/harsh/All%20in%20one/Quant%20Resources/AZ%20Quant%20Project/quant-deep-hedging/notebooks)**:
    *   `01_bs_baseline.ipynb`: Baseline Black-Scholes delta-hedging (adjusted for futures contracts, $r=0.0$).
    *   `02_adversarial_deep_hedging.ipynb`: Main deep hedging experiments and visualizations.

---

## Backtest & Simulation Results

Option hedging performance evaluated on 10,000 Monte Carlo paths ($N=20$, $c=10$ bps):

### 1. Standard Market (GBM Paths)
Under standard log-normal assumptions:

| Hedging Strategy | Mean P&L | Std Dev P&L | CVaR (95%) | Downside Risk | Hedging Error (RMSE) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Black-Scholes (BS)** | -0.0824 | 0.0154 | 0.1186 | 0.0154 | 0.0838 |
| **Standard Deep Hedging** | -0.0605 | 0.0912 | 0.3201 | 0.0912 | 0.1095 |
| **Adversarial Deep Hedging** | -0.0775 | 0.0358 | 0.1603 | 0.0358 | 0.0854 |

### 2. Stressed Market (Regime-Switching Paths)
Under regime-switching volatility spikes and market selloffs:

| Hedging Strategy | Mean P&L | Std Dev P&L | CVaR (95%) | Downside Risk | Hedging Error (RMSE) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Black-Scholes (BS)** | -0.0917 | 0.0432 | 0.2198 | 0.0432 | 0.1014 |
| **Standard Deep Hedging** | -0.0950 | 0.1339 | 0.4781 | 0.1339 | 0.1642 |
| **Adversarial Deep Hedging** | -0.0917 | 0.0549 | 0.2194 | 0.0549 | 0.1070 |

### Key Observations:
1.  **Regime Switch Robustness**: Standard deep hedging (trained only on normal GBM paths) breaks down under market stress, causing its tail loss (CVaR) to spike to **0.4781**. The **Adversarial Deep Hedger** limits CVaR to **0.2194**, showing strong robustness.
2.  **No-Transaction Band**: Under transaction costs, the neural network models learn a smoother delta profile relative to Black-Scholes (reducing trade frequency on small spot movements) to optimize P&L.

---

## Setup & Running the Notebooks
Install dependencies:
```bash
pip install torch pandas numpy matplotlib jupyter
```
Run Jupyter:
```bash
jupyter notebook
```