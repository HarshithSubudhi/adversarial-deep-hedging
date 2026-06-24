import numpy as np
import torch

def simulate_gbm(S0=1.0, mu=0.05, sigma=0.20, T=1.0, N=20, num_paths=10000):
    """Simulates stock price paths using standard Geometric Brownian Motion (GBM).
    
    Returns:
        paths: NumPy array of shape (N + 1, num_paths)
    """
    dt = T / N
    paths = np.zeros((N + 1, num_paths))
    paths[0] = S0
    
    for i in range(1, N + 1):
        # Draw independent normal random variables
        Z = np.random.normal(0, 1, num_paths)
        paths[i] = paths[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
    return paths

def simulate_stressed_gbm(S0=1.0, T=1.0, N=20, num_paths=10000):
    """Simulates stock price paths using a Markov Regime-Switching stressed model.
    
    Regimes:
        Regime 0 (Normal): Low volatility, slightly positive drift.
        Regime 1 (Stressed): High volatility, negative drift (market crash/stress).
        
    Returns:
        paths: NumPy array of shape (N + 1, num_paths)
    """
    dt = T / N
    paths = np.zeros((N + 1, num_paths))
    paths[0] = S0
    
    # Regime parameters
    mu_normal, sigma_normal = 0.05, 0.15
    mu_stress, sigma_stress = -0.25, 0.45
    
    # Transition probability matrix: P = [[P(0->0), P(0->1)], [P(1->0), P(1->1)]]
    # High probability of staying in same state (normal 95%, stress 85%)
    p_stay_normal = 0.95
    p_stay_stress = 0.85
    
    # Initialize all paths in the normal state (0)
    current_regime = np.zeros(num_paths, dtype=int)
    
    for i in range(1, N + 1):
        Z = np.random.normal(0, 1, num_paths)
        
        # Determine drift and vol for each path based on its current regime
        mu = np.where(current_regime == 0, mu_normal, mu_stress)
        sigma = np.where(current_regime == 0, sigma_normal, sigma_stress)
        
        paths[i] = paths[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        # Transition regime states for the next step
        u = np.random.uniform(0, 1, num_paths)
        next_regime = np.zeros(num_paths, dtype=int)
        
        # For normal paths: transition to stress if u > p_stay_normal
        next_regime = np.where((current_regime == 0) & (u > p_stay_normal), 1, next_regime)
        next_regime = np.where((current_regime == 0) & (u <= p_stay_normal), 0, next_regime)
        
        # For stressed paths: transition to normal if u > p_stay_stress
        next_regime = np.where((current_regime == 1) & (u > p_stay_stress), 0, next_regime)
        next_regime = np.where((current_regime == 1) & (u <= p_stay_stress), 1, next_regime)
        
        current_regime = next_regime
        
    return paths

def simulate_adversarial(generator, S0=1.0, T=1.0, N=20, num_paths=10000):
    """Simulates stock price paths using the trained PyTorch AdversarialGenerator.
    
    Returns:
        paths: NumPy array of shape (N + 1, num_paths)
    """
    generator.eval()
    hidden_dim = generator.hidden_dim
    device = next(generator.parameters()).device
    
    paths = np.zeros((N + 1, num_paths))
    paths[0] = S0
    
    prev_S = torch.full((num_paths, 1), S0, dtype=torch.float32, device=device)
    prev_hidden = torch.zeros((num_paths, hidden_dim), dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for i in range(1, N + 1):
            # Draw standard normal noise
            noise = torch.randn((num_paths, 1), dtype=torch.float32, device=device)
            next_S, next_hidden, _, _ = generator(prev_S, prev_hidden, noise)
            
            paths[i] = next_S.squeeze(-1).cpu().numpy()
            prev_S = next_S
            prev_hidden = next_hidden
            
    return paths
