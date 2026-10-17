"""
Continuous-Time Neural Option Pricer & Volatility Surface Inversion Module.
Prices European vanilla, Asian, and path-dependent Barrier contracts.
Inverts Black-Scholes formulas to construct exact Implied Volatility Smiles.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import scipy.stats as stats
import torch
from .neural_sde import NoArbitrageNeuralSDE
from .stochastic_processes import CorrelatedRoughPathGenerator
from .numerical_integrator import StochasticPathIntegrator


def black_scholes_call_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """Standard Black-Scholes European call pricing formula."""
    if t <= 1e-4 or sigma <= 1e-4:
        return max(0.0, s - k)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return float(s * stats.norm.cdf(d1) - k * np.exp(-r * t) * stats.norm.cdf(d2))


def invert_implied_volatility(
    price: float, s: float, k: float, t: float, r: float, tol: float = 1e-5
) -> float:
    """Numerical inversion using Brent's method to extract implied volatility."""
    intrinsic = max(0.0, s - k * np.exp(-r * t))
    if price <= intrinsic:
        return 0.001

    low, high = 0.001, 3.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        mid_price = black_scholes_call_price(s, k, t, r, mid)
        if abs(mid_price - price) < tol:
            return float(mid)
        if mid_price < price:
            low = mid
        else:
            high = mid
    return float(0.5 * (low + high))


class NeuralPathPricer:
    """
    Evaluates vanilla option surfaces and path-dependent derivatives
    using sample trajectories from the NoArbitrageNeuralSDE.
    """

    def __init__(self, integrator: StochasticPathIntegrator):
        self.integrator = integrator

    def price_european_surface(
        self,
        s_0: float = 100.0,
        strikes: List[float] = None,
        maturities: List[float] = None,
        n_paths: int = 5000,
        n_steps_per_year: int = 100
    ) -> Dict[str, Any]:
        """
        Prices a full grid of European calls and computes the implied volatility surface.
        """
        strikes = strikes or [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
        maturities = maturities or [0.25, 0.50, 1.0]  # 3M, 6M, 1Y
        r = self.integrator.model.r

        call_surface = np.zeros((len(maturities), len(strikes)))
        iv_surface = np.zeros((len(maturities), len(strikes)))

        # Simulate paths for the longest maturity
        max_t = max(maturities)
        total_steps = int(max_t * n_steps_per_year)

        time_grid, s_paths, v_paths = self.integrator.simulate_paths(
            s_0=s_0, t_max=max_t, n_steps=total_steps, n_paths=n_paths
        )
        time_grid_np = time_grid.detach().cpu().numpy()
        s_paths_np = s_paths.detach().cpu().numpy()

        for i, t_mat in enumerate(maturities):
            # Find step index corresponding to t_mat
            step_idx = int(np.argmin(np.abs(time_grid_np - t_mat)))
            terminal_prices = s_paths_np[:, step_idx]
            discount = np.exp(-r * t_mat)

            for j, k in enumerate(strikes):
                payoffs = np.maximum(terminal_prices - k, 0.0)
                call_price = float(discount * np.mean(payoffs))
                call_surface[i, j] = call_price

                # Invert implied volatility
                iv = invert_implied_volatility(call_price, s_0, k, t_mat, r)
                iv_surface[i, j] = iv

        return {
            "s_0": s_0,
            "strikes": strikes,
            "maturities": maturities,
            "call_surface": call_surface,
            "implied_vol_surface": iv_surface,
            "s_paths_sample": s_paths_np[:10, :]  # Sample paths for plotting
        }

    def price_barrier_option(
        self,
        s_0: float = 100.0,
        strike: float = 100.0,
        barrier: float = 125.0,
        t_max: float = 1.0,
        n_paths: int = 5000,
        n_steps: int = 100
    ) -> float:
        """
        Prices an Up-and-Out European Call option:
        Payoff = max(S_T - K, 0) if max_{t in [0,T]} S_t < Barrier, else 0.
        """
        r = self.integrator.model.r
        time_grid, s_paths, _ = self.integrator.simulate_paths(
            s_0=s_0, t_max=t_max, n_steps=n_steps, n_paths=n_paths
        )
        s_np = s_paths.cpu().numpy()

        # Check barrier breach along each path
        max_prices = np.max(s_np, axis=1)
        active_mask = max_prices < barrier

        terminal_s = s_np[:, -1]
        payoffs = np.where(active_mask, np.maximum(terminal_s - strike, 0.0), 0.0)
        price = float(np.exp(-r * t_max) * np.mean(payoffs))
        return price
