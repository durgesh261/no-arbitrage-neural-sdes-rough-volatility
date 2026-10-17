import os
import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stochastic_processes import CorrelatedRoughPathGenerator
from src.neural_sde import NoArbitrageNeuralSDE
from src.numerical_integrator import StochasticPathIntegrator
from src.arbitrage_checker import ArbitrageVerifier


@pytest.fixture
def setup_sde():
    path_gen = CorrelatedRoughPathGenerator(hurst=0.12, rho=-0.70, random_state=42)
    sde_model = NoArbitrageNeuralSDE(risk_free_rate=0.035, rho=-0.70, initial_vol=0.20)
    integrator = StochasticPathIntegrator(model=sde_model, path_generator=path_gen)
    verifier = ArbitrageVerifier()
    return integrator, verifier


def test_martingale_condition_holds(setup_sde):
    """Verifies E^Q[e^(-rT) S_T] == S_0 within Monte Carlo confidence interval."""
    integrator, verifier = setup_sde
    s_0 = 100.0
    r = integrator.model.r
    t_max = 1.0

    time_grid, s_paths, _ = integrator.simulate_paths(s_0=s_0, t_max=t_max, n_steps=60, n_paths=3000)
    err_pct, is_valid = verifier.verify_martingale_condition(s_paths, s_0, r, t_max)

    assert is_valid is True
    assert err_pct < 2.0  # discrepancy under 2%


def test_volatility_positivity(setup_sde):
    """Ensures neural volatility surface remains strictly positive everywhere."""
    integrator, _ = setup_sde
    sde = integrator.model

    log_m = torch.linspace(-0.5, 0.5, 50).view(-1, 1)
    v = torch.full((50, 1), 0.04)
    t = torch.full((50, 1), 0.5)

    sigma = sde.vol_network(log_m, v, t)
    assert torch.all(sigma >= 0.01)
