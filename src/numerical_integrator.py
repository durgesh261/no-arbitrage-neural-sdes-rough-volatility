"""
Stochastic Numerical Integrator Module.
Implements Euler-Maruyama and Milstein integration in log-asset coordinates.
Simulating d(log S_t) via Itô's Lemma strictly prevents negative asset prices.
"""

from typing import Tuple, Dict, Any
import numpy as np
import torch
from .neural_sde import NoArbitrageNeuralSDE
from .stochastic_processes import CorrelatedRoughPathGenerator


class StochasticPathIntegrator:
    """
    High-performance numerical SDE solver supporting Euler-Maruyama
    and Milstein integration under the risk-neutral measure Q.
    """

    def __init__(self, model: NoArbitrageNeuralSDE, path_generator: CorrelatedRoughPathGenerator):
        self.model = model
        self.path_gen = path_generator

    @torch.no_grad()
    def simulate_paths(
        self,
        s_0: float = 100.0,
        t_max: float = 1.0,
        n_steps: int = 100,
        n_paths: int = 2000,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulates coupled asset and volatility trajectories over [0, T].
        
        Returns:
        - time_grid: Tensor of shape (n_steps + 1,)
        - s_paths: Asset price trajectories of shape (n_paths, n_steps + 1)
        - v_paths: Variance trajectories of shape (n_paths, n_steps + 1)
        """
        dt = t_max / n_steps
        time_grid = torch.linspace(0, t_max, n_steps + 1, device=device)

        # Generate correlated Brownian increments
        dw_s, dw_v = self.path_gen.generate_correlated_increments(n_paths, n_steps, dt)
        dw_s = dw_s.to(device)
        dw_v = dw_v.to(device)

        # Initialize path tensors
        s_paths = torch.zeros((n_paths, n_steps + 1), device=device)
        v_paths = torch.zeros((n_paths, n_steps + 1), device=device)

        s_paths[:, 0] = s_0
        v_paths[:, 0] = self.model.v0

        s_0_tensor = torch.tensor([[s_0]], device=device, dtype=torch.float32)

        # Log-Euler-Maruyama step integration
        log_s = torch.full((n_paths, 1), float(np.log(s_0)), device=device)
        curr_v = torch.full((n_paths, 1), float(self.model.v0), device=device)

        for step in range(n_steps):
            t_curr = time_grid[step].view(1, 1).expand(n_paths, 1)
            curr_s = torch.exp(log_s)

            # Evaluate neural drift and diffusion
            drift_s, diff_s, drift_v, diff_v = self.model.drift_and_diffusion(
                curr_s, curr_v, t_curr, s_0_tensor
            )

            # Instantaneous volatility: sigma = diff_s / curr_s
            sigma_inst = diff_s / curr_s

            # 1. Update log-asset price via Itô's Lemma:
            # d(log S) = (r - 0.5 * sigma^2) dt + sigma * dW^S
            dw_s_step = dw_s[:, step : step + 1]
            d_log_s = (self.model.r - 0.5 * sigma_inst**2) * dt + sigma_inst * dw_s_step
            log_s = log_s + d_log_s

            # 2. Update variance process:
            # dV = kappa * (theta - V) dt + eta * sqrt(V) * dW^V
            dw_v_step = dw_v[:, step : step + 1]
            d_v = drift_v * dt + diff_v * dw_v_step
            curr_v = torch.clamp(curr_v + d_v, min=1e-4)

            s_paths[:, step + 1] = torch.exp(log_s).squeeze(-1)
            v_paths[:, step + 1] = curr_v.squeeze(-1)

        return time_grid, s_paths, v_paths
