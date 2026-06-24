import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type):
    """Vectorized Black-Scholes model.
    
    Returns:
        price: Option price (float or ndarray)
        delta: Option Delta (float or ndarray)
        gamma: Option Gamma (float or ndarray)
        theta: Option Theta (float or ndarray)
    """
    # Handle time to expiry edge cases
    T = np.maximum(T, 1e-8)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    
    return price, delta, gamma, theta

def implied_volatility(observed_price, S, K, T, r, option_type, tol=1e-6, max_iter=100):
    """Computes Black-Scholes implied volatility using a bisection search."""
    low_vol = 1e-4
    high_vol = 5.0
    
    for _ in range(max_iter):
        mid_vol = (low_vol + high_vol) / 2.0
        # Correctly unpack 4 values returned by black_scholes
        price, _, _, _ = black_scholes(S, K, T, r, mid_vol, option_type)
        
        if abs(price - observed_price) < tol:
            return mid_vol
            
        if price < observed_price:
            low_vol = mid_vol
        else:
            high_vol = mid_vol
            
    return (low_vol + high_vol) / 2.0

def calculate_cvar(pnl, alpha=0.95):
    """Calculates Conditional Value-at-Risk (expected tail loss) for P&L distribution."""
    losses = -pnl
    var = np.percentile(losses, alpha * 100)
    return np.mean(losses[losses >= var])

def calculate_pnl(delta_sequence, S_path, K, option_type):
    """Calculates P&L: -Payoff + Trading Profit."""
    price_changes = np.diff(S_path)
    # Trading P&L: Sum of (delta_t * dS_t)
    trading_pnl = np.sum(delta_sequence[:-1] * price_changes)
    
    S_T = S_path[-1]
    payoff = max(0, S_T - K) if option_type == 'call' else max(0, K - S_T)
    return -payoff + trading_pnl