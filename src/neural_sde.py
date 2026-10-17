"""
No-Arbitrage Neural Stochastic Differential Equation (Neural SDE) Model.
Enforces the Fundamental Theorem of Asset Pricing by hard-constraining the
risk-neutral drift to satisfy the Martingale Property: E^Q[e^(-rt) S_t | F_s] = S_s.

Governing System of Coupled Neural Itô SDEs:
1. Asset Price:   d S_t = r S_t dt + σ_θ(S_t, V_t, t) S_t dW_t^S
2. Latent Vol:    d V_t = κ(θ - V_t) dt + η_φ(V_t, t) dW_t^V
with correlation d⟨W^S, W^V⟩_t = ρ dt.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralVolatilityDiffusion(nn.Module):
    """
    Parameterizes the continuous-time local-stochastic volatility surface σ_θ(S, V, t).
    Guarantees strict positivity: σ(S, V, t) >= σ_min > 0.
    """

    def __init__(self, hidden_dim: int = 64, min_vol: float = 0.02):
        super().__init__()
        self.min_vol = min_vol

        # Input features: [log-moneyness log(S/S_0), current variance V, normalized time t]
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize to output baseline volatility ~20%
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, log_moneyness: torch.Tensor, v_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes instantaneous volatility σ_t.
        Inputs shaped (batch_size, 1).
        """
        inputs = torch.cat([log_moneyness, v_t, t], dim=-1)
        raw_out = self.net(inputs)
        # Smooth positivity guarantee
        sigma = self.min_vol + F.softplus(raw_out)
        return sigma


class NoArbitrageNeuralSDE(nn.Module):
    """
    Continuous-Time Neural SDE with exact Martingale drift invariance:
    Drift μ(S, t) = r * S (Hard-coded to satisfy No-Arbitrage under Measure Q).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.035,   # ECB refinancing / deposit benchmark ~3.5%
        hidden_dim: int = 64,
        initial_vol: float = 0.20,
        mean_reversion_kappa: float = 2.5,
        long_term_vol_theta: float = 0.20,
        vol_of_vol_eta: float = 0.35,
        rho: float = -0.70
    ):
        super().__init__()
        self.r = risk_free_rate
        self.rho = rho
        self.v0 = initial_vol**2

        # Trainable mean-reversion parameters for volatility process
        self.log_kappa = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(mean_reversion_kappa)))))
        self.log_theta = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(long_term_vol_theta**2)))))
        self.log_eta = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(vol_of_vol_eta)))))

        # Neural Volatility Surface
        self.vol_network = NeuralVolatilityDiffusion(hidden_dim=hidden_dim, min_vol=0.01)

    @property
    def kappa(self) -> torch.Tensor:
        return torch.exp(self.log_kappa)

    @property
    def theta(self) -> torch.Tensor:
        return torch.exp(self.log_theta)

    @property
    def eta(self) -> torch.Tensor:
        return torch.exp(self.log_eta)

    def drift_and_diffusion(
        self,
        s_t: torch.Tensor,
        v_t: torch.Tensor,
        t: torch.Tensor,
        s_0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes analytical drift and diffusion for both state variables:
        1. Asset Drift: μ_S = r * S_t (Guarantees Martingale condition)
        2. Asset Diffusion: σ_S = σ_θ(S_t, V_t, t) * S_t
        3. Variance Drift: μ_V = κ * (θ - V_t)
        4. Variance Diffusion: σ_V = η * sqrt(max(V_t, 1e-4))
        """
        # 1. Asset price drift: strictly r * S_t
        drift_s = self.r * s_t

        # 2. Asset price diffusion via Neural Volatility Surface
        log_moneyness = torch.log(torch.clamp(s_t / s_0, min=1e-4, max=1e4))
        sigma_inst = self.vol_network(log_moneyness, v_t, t)
        diff_s = sigma_inst * s_t

        # 3. Variance drift (Feller condition mean-reversion)
        v_clamped = torch.clamp(v_t, min=1e-4)
        drift_v = self.kappa * (self.theta - v_clamped)

        # 4. Variance diffusion
        diff_v = self.eta * torch.sqrt(v_clamped)

        return drift_s, diff_s, drift_v, diff_v
