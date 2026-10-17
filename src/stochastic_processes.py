"""
Stochastic Processes & Rough Fractional Brownian Motion Generator.
Implements non-Markovian path generation with Hurst parameter H in (0.05, 0.3)
consistent with empirical German DAX & European market rough volatility stylized facts.
Reference: Gatheral, Jaisson, Rosenbaum (2018) "Volatility is rough".
"""

from typing import Tuple, Optional
import numpy as np
import torch


class CorrelatedRoughPathGenerator:
    """
    Generates coupled Brownian increments (dW_t^S, dW_t^V) with leverage correlation rho,
    as well as fractional Brownian motion (fBM) increments with Hurst parameter H < 0.5.
    """

    def __init__(self, hurst: float = 0.12, rho: float = -0.72, random_state: int = 42):
        """
        Parameters:
        - hurst: Hurst parameter (H < 0.5 corresponds to rough, anti-persistent regime).
        - rho: Correlation between asset price returns and volatility shocks (leverage effect).
        """
        if not (0.01 < hurst < 0.50):
            raise ValueError(f"Hurst parameter must be in (0.01, 0.50) for rough paths, got {hurst}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"Correlation rho must be in [-1, 1], got {rho}")

        self.hurst = hurst
        self.rho = rho
        self.rng = np.random.default_rng(random_state)

    def generate_correlated_increments(
        self, n_paths: int, n_steps: int, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates standard 2D Brownian increments (dW_S, dW_V) with correlation rho:
        dW_S = sqrt(dt) * Z_1
        dW_V = sqrt(dt) * (rho * Z_1 + sqrt(1 - rho^2) * Z_2)
        """
        z1 = self.rng.standard_normal(size=(n_paths, n_steps))
        z2 = self.rng.standard_normal(size=(n_paths, n_steps))

        dw_s = np.sqrt(dt) * z1
        dw_v = np.sqrt(dt) * (self.rho * z1 + np.sqrt(1.0 - self.rho**2) * z2)

        return (
            torch.tensor(dw_s, dtype=torch.float32),
            torch.tensor(dw_v, dtype=torch.float32)
        )

    def generate_fractional_brownian_paths(
        self, n_paths: int, n_steps: int, t_max: float
    ) -> torch.Tensor:
        """
        Simulates fractional Brownian motion B_H(t) with covariance:
        E[B_H(t) B_H(s)] = 0.5 * (t^(2H) + s^(2H) - |t - s|^(2H))
        Using Cholesky decomposition of the exact autocovariance matrix.
        """
        time_grid = np.linspace(0, t_max, n_steps + 1)
        cov = np.zeros((n_steps + 1, n_steps + 1))
        h2 = 2.0 * self.hurst

        for i in range(n_steps + 1):
            for j in range(i, n_steps + 1):
                t_i, t_j = time_grid[i], time_grid[j]
                val = 0.5 * (t_i**h2 + t_j**h2 - abs(t_i - t_j)**h2)
                cov[i, j] = val
                cov[j, i] = val

        # Regularize diagonal for numerical stability of Cholesky
        cov += np.eye(n_steps + 1) * 1e-8
        l_mat = np.linalg.cholesky(cov)

        std_normals = self.rng.standard_normal(size=(n_steps + 1, n_paths))
        fbm_paths = l_mat @ std_normals  # (n_steps + 1, n_paths)

        # Transpose to (n_paths, n_steps + 1)
        fbm_tensor = torch.tensor(fbm_paths.T, dtype=torch.float32)
        return fbm_tensor
