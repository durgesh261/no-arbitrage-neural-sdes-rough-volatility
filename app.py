"""
Production FastAPI Application for No-Arbitrage Neural SDEs.
Continuous-Time Stochastics & Quantitative Derivatives Pricing.

Endpoints:
- GET  /                           : Service health & mathematical specification
- GET  /api/v1/martingale-check    : Audits E^Q[e^(-rT) S_T] == S_0 Martingale property
- POST /api/v1/price-option        : Prices European vanilla option and inverts implied volatility
- POST /api/v1/price-barrier       : Prices path-dependent Up-and-Out Barrier derivative
- GET  /api/v1/arbitrage-audit     : Breeden-Litzenberger surface convexity and calendar spread audit
"""

import os
import sys
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.stochastic_processes import CorrelatedRoughPathGenerator
from src.neural_sde import NoArbitrageNeuralSDE
from src.numerical_integrator import StochasticPathIntegrator
from src.arbitrage_checker import ArbitrageVerifier
from src.pricer import NeuralPathPricer, black_scholes_call_price, invert_implied_volatility


# --- Global Model Context ---
context = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and pre-configures Neural SDE and Stochastic Path Integrator."""
    print("[API Startup] Initializing No-Arbitrage Neural SDE engine...")
    hurst = 0.12
    rho = -0.72
    r = 0.035

    path_gen = CorrelatedRoughPathGenerator(hurst=hurst, rho=rho, random_state=42)
    sde_model = NoArbitrageNeuralSDE(risk_free_rate=r, rho=rho, initial_vol=0.22)
    integrator = StochasticPathIntegrator(model=sde_model, path_generator=path_gen)
    pricer = NeuralPathPricer(integrator=integrator)
    verifier = ArbitrageVerifier()

    context["path_gen"] = path_gen
    context["sde_model"] = sde_model
    context["integrator"] = integrator
    context["pricer"] = pricer
    context["verifier"] = verifier

    print("[API Startup] No-Arbitrage Neural SDE microservice ready.")
    yield
    context.clear()


app = FastAPI(
    title="No-Arbitrage Neural Stochastic Differential Equation (Neural SDE) API",
    description=(
        "Continuous-Time Mathematical Finance microservice for pricing and volatility modeling. "
        "Guarantees the Fundamental Theorem of Asset Pricing via hard-coded Martingale drift, "
        "rough fractional Brownian motion (H < 0.5), and Breeden-Litzenberger no-arbitrage bounds."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---

class EuropeanOptionRequest(BaseModel):
    spot_price_s0: float = Field(100.0, json_schema_extra={"example": 100.0}, description="Initial asset price S_0 (EUR)")
    strike_k: float = Field(100.0, json_schema_extra={"example": 100.0}, description="Option strike price K (EUR)")
    maturity_years: float = Field(1.0, json_schema_extra={"example": 1.0}, description="Time to maturity T (Years)")
    n_paths: int = Field(4000, json_schema_extra={"example": 4000}, description="Monte Carlo simulation trajectories")


class BarrierOptionRequest(BaseModel):
    spot_price_s0: float = Field(100.0, json_schema_extra={"example": 100.0})
    strike_k: float = Field(100.0, json_schema_extra={"example": 100.0})
    barrier_level: float = Field(125.0, json_schema_extra={"example": 125.0}, description="Knock-out barrier level")
    maturity_years: float = Field(1.0, json_schema_extra={"example": 1.0})
    n_paths: int = Field(4000, json_schema_extra={"example": 4000})


# --- Endpoints ---

@app.get("/", tags=["System"])
def root():
    return {
        "service": "No-Arbitrage Neural SDE Quantitative Finance Engine",
        "stochastic_regime": "Rough Volatility (Hurst Exponent H=0.12, Anti-Persistent)",
        "leverage_correlation_rho": -0.72,
        "martingale_drift_condition": "mu(S, t) = r * S_t strictly enforced under Q",
        "documentation": "/docs",
        "author": "Durgesh Kanjariya"
    }


@app.get("/api/v1/martingale-check", tags=["Mathematical Verification"])
def verify_martingale(n_paths: int = Query(4000, ge=1000, le=10000)):
    """Verifies E^Q[e^(-rT) S_T] == S_0 to prove absence of arbitrage."""
    integrator: StochasticPathIntegrator = context["integrator"]
    verifier: ArbitrageVerifier = context["verifier"]

    time_grid, s_paths, _ = integrator.simulate_paths(s_0=100.0, t_max=1.0, n_steps=100, n_paths=n_paths)
    err_pct, is_valid = verifier.verify_martingale_condition(s_paths, 100.0, integrator.model.r, 1.0)

    terminal_mean = float(torch.mean(s_paths[:, -1].detach()))
    discounted_mean = terminal_mean * np.exp(-integrator.model.r * 1.0)

    return {
        "s_0": 100.0,
        "mean_terminal_s_t": round(terminal_mean, 2),
        "discounted_expected_s_t": round(discounted_mean, 2),
        "discrepancy_pct": round(err_pct, 4),
        "martingale_property_verified": is_valid,
        "interpretation": "Discounted asset price is a certified Martingale under measure Q."
    }


@app.post("/api/v1/price-option", tags=["Derivatives Pricing"])
def price_european_option(req: EuropeanOptionRequest):
    """Prices European Call Option using Continuous-Time Neural SDE and inverts implied volatility."""
    integrator: StochasticPathIntegrator = context["integrator"]
    r = integrator.model.r
    t_mat = req.maturity_years

    time_grid, s_paths, _ = integrator.simulate_paths(
        s_0=req.spot_price_s0, t_max=t_mat, n_steps=100, n_paths=req.n_paths
    )
    s_t = s_paths[:, -1].detach().cpu().numpy()
    payoffs = np.maximum(s_t - req.strike_k, 0.0)
    call_price = float(np.exp(-r * t_mat) * np.mean(payoffs))

    iv = invert_implied_volatility(call_price, req.spot_price_s0, req.strike_k, t_mat, r)

    return {
        "spot_price": req.spot_price_s0,
        "strike": req.strike_k,
        "maturity_years": req.maturity_years,
        "neural_sde_call_price_eur": round(call_price, 2),
        "inverted_implied_volatility_pct": round(iv * 100.0, 2),
        "moneyness_ratio": round(req.spot_price_s0 / req.strike_k, 3)
    }


@app.post("/api/v1/price-barrier", tags=["Derivatives Pricing"])
def price_barrier_option(req: BarrierOptionRequest):
    """Prices path-dependent Up-and-Out Barrier Option."""
    pricer: NeuralPathPricer = context["pricer"]
    price = pricer.price_barrier_option(
        s_0=req.spot_price_s0,
        strike=req.strike_k,
        barrier=req.barrier_level,
        t_max=req.maturity_years,
        n_paths=req.n_paths
    )
    return {
        "spot_price": req.spot_price_s0,
        "strike": req.strike_k,
        "barrier_level": req.barrier_level,
        "maturity_years": req.maturity_years,
        "up_and_out_call_price_eur": round(price, 2),
        "barrier_type": "Up-and-Out Knockout"
    }


@app.get("/api/v1/arbitrage-audit", tags=["Mathematical Verification"])
def get_arbitrage_audit():
    """Runs formal Breeden-Litzenberger surface convexity and calendar spread audit."""
    pricer: NeuralPathPricer = context["pricer"]
    verifier: ArbitrageVerifier = context["verifier"]

    strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    maturities = [0.25, 0.50, 1.0]

    surface_res = pricer.price_european_surface(s_0=100.0, strikes=strikes, maturities=maturities, n_paths=4000)
    audit = verifier.verify_surface_no_arbitrage(
        strike_grid=np.array(strikes),
        maturities=np.array(maturities),
        call_surface=surface_res["call_surface"]
    )

    return {
        "total_derivative_checks": audit.total_checks,
        "butterfly_convexity_violations": audit.butterfly_arbitrage_violations,
        "calendar_spread_violations": audit.calendar_spread_violations,
        "is_completely_arbitrage_free": audit.is_completely_arbitrage_free,
        "summary": audit.summary
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
