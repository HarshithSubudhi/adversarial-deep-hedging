import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from engine import compute_torch_pnl

def loss_erm(pnl, risk_lambda):
    """Numerically stable Entropic Risk Measure (ERM) loss.
    
    Computes (1/lambda) * log( Mean( exp(-lambda * pnl) ) ) using the log-sum-exp trick.
    """
    val = -risk_lambda * pnl
    # logsumexp(val) - log(batch_size) = log(sum(exp(val)) / batch_size)
    log_mean_exp = torch.logsumexp(val, dim=0) - torch.log(torch.tensor(float(val.size(0)), device=val.device))
    return log_mean_exp / risk_lambda

def train_standard_hedging(hedger, paths, K, option_type='call', c=0.001, 
                           loss_type='erm', risk_lambda=1.0, alpha=0.95,
                           epochs=20, batch_size=256, lr=1e-3):
    """Trains a DeepHedger on a fixed set of simulated stock paths.
    
    Args:
        paths: NumPy array of shape (N + 1, num_paths)
    """
    device = next(hedger.parameters()).device
    hedger.train()
    
    # Setup CVaR parameter if needed
    var_q = nn.Parameter(torch.tensor(0.0, device=device))
    
    # Define optimizer
    params = list(hedger.parameters())
    if loss_type == 'cvar':
        params.append(var_q)
    optimizer = optim.Adam(params, lr=lr)
    
    num_paths = paths.shape[1]
    N = paths.shape[0] - 1
    dt = 1.0 / N
    
    # Training Loop
    for epoch in range(epochs):
        permutation = np.random.permutation(num_paths)
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, num_paths, batch_size):
            indices = permutation[i:i+batch_size]
            batch_paths = torch.tensor(paths[:, indices], dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            
            pnl = compute_torch_pnl(hedger, batch_paths, K, option_type, c, dt)
            
            if loss_type == 'erm':
                loss = loss_erm(pnl, risk_lambda)
            elif loss_type == 'cvar':
                loss = var_q + (1.0 / (1.0 - alpha)) * torch.mean(torch.clamp(-pnl - var_q, min=0.0))
            else:
                loss = torch.mean(pnl**2) # Fallback to MSE of replication
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        # print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/batches:.6f}")
        
    return hedger


def train_adversarial_hedging(hedger, generator, S0=1.0, K=1.0, T=1.0, N=20,
                              option_type='call', c=0.001, loss_type='erm',
                              risk_lambda=1.0, alpha=0.95, epochs=10, 
                              steps_per_epoch=100, batch_size=256, lr=1e-3):
    """Runs min-max alternating training between DeepHedger and AdversarialGenerator.
    
    Ratio of updates: Hedger (H) is updated 5 times for every 1 update of the Generator (G).
    """
    device = next(hedger.parameters()).device
    
    var_q = nn.Parameter(torch.tensor(0.0, device=device))
    
    hedger_params = list(hedger.parameters())
    if loss_type == 'cvar':
        hedger_params.append(var_q)
        
    opt_H = optim.Adam(hedger_params, lr=lr)
    opt_G = optim.Adam(generator.parameters(), lr=lr)
    
    dt = T / N
    hidden_dim = generator.hidden_dim
    
    for epoch in range(epochs):
        epoch_loss_H = 0.0
        epoch_loss_G = 0.0
        
        for step in range(steps_per_epoch):
            # --- 1. Train the Hedger (H) for 5 iterations ---
            hedger.train()
            generator.eval()
            
            for _ in range(5):
                # Generate paths differentiably from G
                batch_paths_list = [torch.full((batch_size, 1), S0, device=device)]
                prev_S = batch_paths_list[0]
                prev_hidden = torch.zeros((batch_size, hidden_dim), device=device)
                
                for _ in range(N):
                    noise = torch.randn((batch_size, 1), device=device)
                    # Autoregressive generation
                    next_S, next_hidden, _, _ = generator(prev_S, prev_hidden, noise)
                    batch_paths_list.append(next_S)
                    prev_S = next_S
                    prev_hidden = next_hidden
                    
                # Stack to shape (N + 1, batch_size)
                batch_paths = torch.stack(batch_paths_list, dim=0).squeeze(-1)
                
                opt_H.zero_grad()
                pnl = compute_torch_pnl(hedger, batch_paths, K, option_type, c, dt)
                
                if loss_type == 'erm':
                    loss_H = loss_erm(pnl, risk_lambda)
                elif loss_type == 'cvar':
                    loss_H = var_q + (1.0 / (1.0 - alpha)) * torch.mean(torch.clamp(-pnl - var_q, min=0.0))
                else:
                    loss_H = torch.mean(pnl**2)
                    
                loss_H.backward()
                opt_H.step()
                epoch_loss_H += loss_H.item()
                
            # --- 2. Train the Generator (G) for 1 iteration ---
            hedger.eval()
            generator.train()
            
            # Generate paths differentiably from G
            batch_paths_list = [torch.full((batch_size, 1), S0, device=device)]
            prev_S = batch_paths_list[0]
            prev_hidden = torch.zeros((batch_size, hidden_dim), device=device)
            
            for _ in range(N):
                noise = torch.randn((batch_size, 1), device=device)
                next_S, next_hidden, _, _ = generator(prev_S, prev_hidden, noise)
                batch_paths_list.append(next_S)
                prev_S = next_S
                prev_hidden = next_hidden
                
            batch_paths = torch.stack(batch_paths_list, dim=0).squeeze(-1)
            
            opt_G.zero_grad()
            pnl = compute_torch_pnl(hedger, batch_paths, K, option_type, c, dt)
            
            if loss_type == 'erm':
                loss_H = loss_erm(pnl, risk_lambda)
            elif loss_type == 'cvar':
                loss_H = var_q + (1.0 / (1.0 - alpha)) * torch.mean(torch.clamp(-pnl - var_q, min=0.0))
            else:
                loss_H = torch.mean(pnl**2)
                
            # Generator aims to MAXIMIZE Hedger's loss (minimize utility / make paths harder)
            # Therefore, G's objective is the negative of the Hedger's loss
            loss_G = -loss_H
            
            loss_G.backward()
            opt_G.step()
            epoch_loss_G += loss_G.item()
            
        # print(f"Adversarial Epoch {epoch+1}/{epochs} - H Loss: {epoch_loss_H/(5*steps_per_epoch):.6f} - G Loss: {epoch_loss_G/steps_per_epoch:.6f}")
        
    return hedger, generator
