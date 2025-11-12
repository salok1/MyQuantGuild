"""
Black-Scholes pricing and implied volatility utilities.
"""
from __future__ import annotations

import math

import torch

torch.set_default_dtype(torch.float64)


def _to_tensor(value) -> torch.Tensor:
    """Convert scalars or tensors to torch.float64 tensors."""
    return torch.as_tensor(value, dtype=torch.float64)


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal cumulative distribution via the error function."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs_call_torch(S0, K, T, r, q, vol) -> torch.Tensor:
    """
    Black-Scholes-Merton European call price.

    Parameters mirror the traditional formula.  The implementation explicitly
    guards against zero time-to-maturity or volatility by using a small epsilon.
    """
    S0 = _to_tensor(S0)
    K = _to_tensor(K)
    T = torch.clamp(_to_tensor(T), min=1e-12)
    r = _to_tensor(r)
    q = _to_tensor(q)
    vol = torch.clamp(_to_tensor(vol), min=1e-12)

    sqrtT = torch.sqrt(T)
    d1 = (torch.log(S0 / K) + (r - q + 0.5 * vol**2) * T) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT
    discounted_S0 = S0 * torch.exp(-q * T)
    discounted_K = K * torch.exp(-r * T)
    return discounted_S0 * _norm_cdf(d1) - discounted_K * _norm_cdf(d2)


def iv_call_brent_torch(
    C,
    S0,
    K,
    T,
    r,
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> torch.Tensor:
    """
    Solve for the Black-Scholes implied volatility using a robust bisection.

    The solver is intentionally simple and prioritizes clarity.  It searches the
    root of f(vol) = BS(vol) - C on a bounded interval that is expanded until
    the call price is bracketed.
    """
    S0_t = _to_tensor(S0)
    K_t = _to_tensor(K)
    T_t = torch.clamp(_to_tensor(T), min=1e-12)
    r_t = _to_tensor(r)
    q_t = _to_tensor(q)
    price = torch.clamp(_to_tensor(C), min=0.0)

    intrinsic = torch.clamp(S0_t * torch.exp(-q_t * T_t) - K_t * torch.exp(-r_t * T_t), min=0.0)
    price = torch.clamp(price, min=intrinsic, max=S0_t)
    target_price = float(price)

    def price_at(vol: float) -> float:
        return float(bs_call_torch(S0_t, K_t, T_t, r_t, q_t, vol))

    low = 1e-6
    high = 1.0
    while price_at(high) < target_price and high < 10.0:
        high *= 2.0

    root = high
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        price_mid = price_at(mid)
        root = mid
        if abs(price_mid - target_price) < tol:
            break
        if price_mid > target_price:
            high = mid
        else:
            low = mid
    return torch.tensor(root, dtype=torch.float64)
