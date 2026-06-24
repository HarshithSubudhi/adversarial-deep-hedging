import torch
import torch.nn as nn

class DeepHedger(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        """Neural network model for derivative hedging.
        
        Args:
            input_dim: Dimension of input state features [S_t/S_0, delta_{t-1}, t/T].
            hidden_dim: Hidden layer dimension.
        """
        super(DeepHedger, self).__init__()
        # Standard Feedforward Neural Network mapping state to target delta position.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Force output to be in [0, 1] for European Call Option delta.
        )
        
    def forward(self, x):
        """Computes the hedging position.
        
        Args:
            x: Tensor of shape (batch_size, input_dim).
        Returns:
            delta_t: Hedging position in stock of shape (batch_size, 1).
        """
        return self.net(x)


class AdversarialGenerator(nn.Module):
    def __init__(self, hidden_dim=32, noise_dim=1):
        """Recurrent neural network market path generator.
        
        It simulates step-by-step stock price changes by outputting drift
        and bounded volatility parameters.
        
        Args:
            hidden_dim: GRU hidden dimension.
            noise_dim: Random noise input dimension.
        """
        super(AdversarialGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        
        # GRU Cell to carry path memory / state history.
        self.gru_cell = nn.GRUCell(input_size=1 + noise_dim, hidden_size=hidden_dim)
        
        # Output layer producing parameters for log-return calculation:
        # [drift (mu), volatility (sigma)]
        self.param_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, prev_S, prev_hidden, noise):
        """Generates the next price step in an autoregressive fashion.
        
        Args:
            prev_S: Stock price at t_{i-1} of shape (batch_size, 1).
            prev_hidden: Hidden state at t_{i-1} of shape (batch_size, hidden_dim).
            noise: Gaussian noise step at t_i of shape (batch_size, noise_dim).
            
        Returns:
            next_S: Generated stock price at t_i of shape (batch_size, 1).
            next_hidden: Updated hidden state at t_i of shape (batch_size, hidden_dim).
            mu: Output drift of shape (batch_size, 1).
            sigma: Output volatility of shape (batch_size, 1).
        """
        # Feature input combines the previous stock price (normalized to S0=1.0) and random noise.
        # We normalize the price input by log transformation relative to S0 = 1.0
        x = torch.cat([torch.log(prev_S + 1e-8), noise], dim=-1)
        next_hidden = self.gru_cell(x, prev_hidden)
        
        # Estimate return parameters
        params = self.param_net(next_hidden)
        mu = 0.50 * torch.tanh(params[:, 0:1])  # Bound drift to [-0.50, 0.50] to stabilize training
        raw_sigma = params[:, 1:2]
        
        # Bounding volatility between 0.05 and 0.50 using a sigmoid to prevent training divergence.
        sigma = 0.05 + 0.45 * torch.sigmoid(raw_sigma)
        
        # Assume daily time step dt = 1 / 252 (or normalized based on maturity)
        dt = 1.0 / 20.0  # Normalized to maturity of 20 days (20 steps)
        
        # Log-normal evolution: S_t = S_{t-1} * exp( (mu - 0.5 * sigma^2)*dt + sigma*sqrt(dt)*noise )
        # To maintain generation consistency, we use the noise component here.
        log_ret = (mu - 0.5 * sigma**2) * dt + sigma * torch.sqrt(torch.tensor(dt)) * noise
        next_S = prev_S * torch.exp(log_ret)
        
        return next_S, next_hidden, mu, sigma
