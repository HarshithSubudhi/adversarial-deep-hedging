import numpy as np
import torch
from hedging_utils import black_scholes

class DeltaHedgingEngine:
    def __init__(self, option_type='call', K=1.0, r=0.05, T=1.0, c=0.001):
        """Discrete-time option delta hedging and backtesting engine.
        
        Args:
            option_type: 'call', 'put', or 'lookback' (Lookback call option).
            K: Option strike price.
            r: Risk-free rate.
            T: Time to maturity (years).
            c: Proportional transaction cost per unit traded.
        """
        self.option_type = option_type
        self.K = K
        self.r = r
        self.T = T
        self.c = c
        
    def get_payoff(self, S_paths):
        """Calculates option payoff at maturity for each price path.
        
        S_paths shape: (N + 1, num_paths)
        """
        S_T = S_paths[-1]
        if self.option_type == 'call':
            return np.maximum(S_T - self.K, 0)
        elif self.option_type == 'put':
            return np.maximum(self.K - S_T, 0)
        elif self.option_type == 'lookback':
            # Lookback call payoff: max_{t} S_t - K
            max_S = np.max(S_paths, axis=0)
            return np.maximum(max_S - self.K, 0)
        else:
            raise ValueError(f"Unknown option type: {self.option_type}")
            
    def run_bs_hedging(self, S_paths, sigma):
        """Vectorized Black-Scholes delta hedging baseline.
        
        Returns:
            deltas: NumPy array of shape (N, num_paths)
        """
        N, num_paths = S_paths.shape[0] - 1, S_paths.shape[1]
        dt = self.T / N
        deltas = np.zeros((N, num_paths))
        
        for i in range(N):
            S_i = S_paths[i]
            tau_i = np.maximum(self.T - i * dt, 1e-8)
            
            # Compute theoretical delta
            _, delta, _, _ = black_scholes(S_i, self.K, tau_i, self.r, sigma, self.option_type if self.option_type != 'lookback' else 'call')
            deltas[i] = delta
            
        return deltas

    def run_nn_hedging(self, hedger, S_paths):
        """Neural network-based delta hedging evaluation.
        
        Returns:
            deltas: NumPy array of shape (N, num_paths)
        """
        hedger.eval()
        N, num_paths = S_paths.shape[0] - 1, S_paths.shape[1]
        device = next(hedger.parameters()).device
        deltas = np.zeros((N, num_paths))
        
        # Initial position delta_{-1} = 0
        prev_delta = torch.zeros((num_paths, 1), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            for i in range(N):
                S_i = torch.tensor(S_paths[i], dtype=torch.float32, device=device).unsqueeze(-1)
                # Normalize features relative to initial price
                norm_S = S_i / S_paths[0, 0]
                tau_i = torch.full((num_paths, 1), 1.0 - i/N, dtype=torch.float32, device=device)
                
                # Input representation: [normalized stock price, previous position, time to maturity]
                features = torch.cat([norm_S, prev_delta, tau_i], dim=-1)
                
                curr_delta = hedger(features)
                deltas[i] = curr_delta.squeeze(-1).cpu().numpy()
                prev_delta = curr_delta
                
        return deltas

    def evaluate_pnl(self, S_paths, deltas):
        """Evaluates dynamic portfolio values, transaction costs, and final P&L.
        
        Returns:
            pnl: NumPy array of shape (num_paths,) containing terminal net P&L
            trading_returns: NumPy array of shape (num_paths,)
            costs: NumPy array of shape (num_paths,)
        """
        N, num_paths = S_paths.shape[0] - 1, S_paths.shape[1]
        price_changes = np.diff(S_paths, axis=0)
        
        # Trading Returns: Sum of (delta_t * dS_t)
        # deltas shape: (N, num_paths), price_changes shape: (N, num_paths)
        trading_returns = np.sum(deltas * price_changes, axis=0)
        
        # Transaction Costs: Sum of (c * S_t * |delta_t - delta_{t-1}|)
        costs = np.zeros(num_paths)
        # initial cost for setting up delta_0 position
        costs += self.c * S_paths[0] * np.abs(deltas[0])
        
        # intermediate adjustments
        for i in range(1, N):
            costs += self.c * S_paths[i] * np.abs(deltas[i] - deltas[i-1])
            
        # terminal liquidation cost (liquidating delta_{N-1} to zero)
        costs += self.c * S_paths[N] * np.abs(deltas[N-1])
        
        payoff = self.get_payoff(S_paths)
        
        # Short option terminal net P&L: -Payoff + Trading Returns - Costs
        pnl = -payoff + trading_returns - costs
        return pnl, trading_returns, costs

    @staticmethod
    def calculate_metrics(pnl, alpha=0.95):
        """Calculates risk metrics for the terminal P&L distribution.
        
        Returns:
            metrics: Dictionary of calculated values.
        """
        # Hedging cost / loss = -pnl
        losses = -pnl
        mean_pnl = np.mean(pnl)
        std_pnl = np.std(pnl)
        
        # Value-at-Risk (VaR)
        var = np.percentile(losses, alpha * 100)
        # Conditional Value-at-Risk (CVaR)
        cvar = np.mean(losses[losses >= var])
        
        # Downside Risk (standard deviation of negative P&L)
        downside_pnl = pnl[pnl < 0]
        downside_risk = np.std(downside_pnl) if len(downside_pnl) > 0 else 0.0
        
        # Hedging Error defined as the Root Mean Squared Hedging Error (RMSE)
        # where we measure deviation around perfect replication (0 net P&L)
        rmse_error = np.sqrt(np.mean(pnl**2))
        
        return {
            'Mean P&L': mean_pnl,
            'Std Dev P&L': std_pnl,
            'VaR (95%)': var,
            'CVaR (95%)': cvar,
            'Downside Risk': downside_risk,
            'Hedging Error (RMSE)': rmse_error
        }


# --- PyTorch helper functions for model training ---

def compute_torch_pnl(hedger, paths_tensor, K, option_type='call', c=0.001, dt=0.05):
    """Calculates terminal P&L in PyTorch to support gradient backpropagation.
    
    Args:
        hedger: DeepHedger PyTorch model.
        paths_tensor: Tensor of shape (N + 1, batch_size) of generated prices.
        K: Option strike price.
        option_type: 'call' or 'lookback'.
        c: Proportional transaction cost.
        dt: Rebalancing time step.
        
    Returns:
        pnl: Tensor of shape (batch_size,)
    """
    N, batch_size = paths_tensor.shape[0] - 1, paths_tensor.shape[1]
    
    trading_returns = torch.zeros(batch_size, device=paths_tensor.device)
    costs = torch.zeros(batch_size, device=paths_tensor.device)
    
    prev_delta = torch.zeros((batch_size, 1), device=paths_tensor.device)
    
    for i in range(N):
        S_i = paths_tensor[i].unsqueeze(-1)
        norm_S = S_i / paths_tensor[0, 0]
        tau_i = torch.full((batch_size, 1), 1.0 - i/N, device=paths_tensor.device)
        
        # Compute delta
        features = torch.cat([norm_S, prev_delta, tau_i], dim=-1)
        curr_delta = hedger(features)
        
        # Add trading returns: delta_t * dS_t
        dS = paths_tensor[i+1] - paths_tensor[i]
        trading_returns += curr_delta.squeeze(-1) * dS
        
        # Add transaction cost: c * S_t * |delta_t - delta_{t-1}|
        costs += c * paths_tensor[i] * torch.abs(curr_delta.squeeze(-1) - prev_delta.squeeze(-1))
        
        prev_delta = curr_delta
        
    # Terminal liquidation cost
    costs += c * paths_tensor[N] * torch.abs(prev_delta.squeeze(-1))
    
    # Calculate payoff
    S_T = paths_tensor[-1]
    if option_type == 'call':
        payoff = torch.clamp(S_T - K, min=0.0)
    elif option_type == 'lookback':
        max_S = torch.max(paths_tensor, dim=0).values
        payoff = torch.clamp(max_S - K, min=0.0)
    else:
        payoff = torch.clamp(S_T - K, min=0.0)
        
    pnl = -payoff + trading_returns - costs
    return pnl
