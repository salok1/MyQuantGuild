"""
Heston characteristic function and Carr-Madan FFT pricing utilities.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


@dataclass
class HestonParams:
    """Container for calibrated Heston parameters."""

    kappa: torch.Tensor
    theta: torch.Tensor
    sigma: torch.Tensor
    rho: torch.Tensor
    v0: torch.Tensor

    @classmethod
    def from_unconstrained(
        cls,
        raw_kappa: torch.Tensor,
        raw_theta: torch.Tensor,
        raw_sigma: torch.Tensor,
        raw_rho: torch.Tensor,
        raw_v0: torch.Tensor,
    ) -> "HestonParams":
        """
        Map unconstrained neural-network outputs to physically valid parameters.
        """
        softplus = lambda x: F.softplus(x) + 1e-6  # ensure strict positivity
        rho = torch.tanh(raw_rho)  # maps to (-1, 1)
        return cls(
            kappa=softplus(raw_kappa),
            theta=softplus(raw_theta),
            sigma=softplus(raw_sigma),
            rho=rho,
            v0=softplus(raw_v0),
        )


def _to_tensor(value, dtype=torch.float64):
    return torch.as_tensor(value, dtype=dtype)


def heston_cf(u, T, S0, r, q, params: HestonParams) -> torch.Tensor:
    """
    Characteristic function of log spot under the risk-neutral Heston model
    using the Little Heston Trap formulation.
    """
    complex_dtype = torch.complex128
    u = torch.as_tensor(u, dtype=complex_dtype)
    T = _to_tensor(T)
    S0 = _to_tensor(S0)
    r = _to_tensor(r)
    q = _to_tensor(q)

    kappa = params.kappa.to(T)
    theta = params.theta.to(T)
    sigma = params.sigma.to(T)
    rho = params.rho.to(T)
    v0 = params.v0.to(T)

    i = torch.complex(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)).to(complex_dtype)
    a = kappa * theta
    eps = torch.tensor(1e-12, dtype=torch.float64, device=T.device)

    beta = kappa - rho * sigma * i * u
    d = torch.sqrt(beta * beta + (sigma**2) * (i * u + u * u))
    g = (beta - d) / (beta + d + eps)

    exp_term = torch.exp(-d * T)
    C = (r - q) * i * u * T + (a / (sigma**2)) * ((beta - d) * T - 2.0 * torch.log((1 - g * exp_term) / (1 - g + eps)))
    D = ((beta - d) / (sigma**2)) * ((1 - exp_term) / (1 - g * exp_term + eps))
    log_S0 = torch.log(S0 * torch.exp(-q * T))
    return torch.exp(C + D * v0 + i * u * log_S0)


def carr_madan_call_torch(
    S0,
    r,
    q,
    T,
    params: HestonParams,
    K,
    alpha: float = 1.5,
    Nfft: int = 2**12,
    eta: float = 0.25,
) -> torch.Tensor:
    """
    Compute European call prices via the Carr-Madan FFT algorithm.

    The implementation keeps the numerical steps explicit for readability and to
    ease future adjustments.  All tensors live in torch.float64 / torch.complex128
    to maintain stability for gradients.
    """
    device = torch.device("cpu")
    dtype = torch.float64
    complex_dtype = torch.complex128

    S0 = _to_tensor(S0).to(dtype)
    r = _to_tensor(r).to(dtype)
    q = _to_tensor(q).to(dtype)
    T = _to_tensor(T).to(dtype)
    K = torch.as_tensor(K, dtype=dtype)
    alpha = torch.tensor(alpha, dtype=dtype)
    eta = torch.tensor(eta, dtype=dtype)

    j = torch.arange(Nfft, dtype=dtype, device=device)
    v = j * eta
    i = torch.complex(torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)).to(complex_dtype)

    u = v - (alpha + 1.0) * i
    cf_vals = heston_cf(u, T, S0, r, q, params)

    numerator = torch.exp(-r * T) * cf_vals
    denominator = alpha * alpha + alpha - v * v + i * (2.0 * alpha + 1.0) * v
    integrand = numerator / (denominator + 1e-12)

    # Simpson's rule weights: 1,4,2,4,...,2,4,1
    weights = torch.ones_like(v)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    weights = weights * eta / 3.0

    delta_k = 2.0 * torch.pi / (Nfft * eta)
    k_grid = -0.5 * Nfft * delta_k + j * delta_k
    b = 0.5 * Nfft * delta_k

    fft_input = integrand * weights * torch.exp(i * v * b)
    fft_values = torch.fft.fft(fft_input)
    call_grid = torch.exp(-alpha * k_grid) / torch.pi * torch.real(fft_values)

    log_strikes = torch.log(K)
    idx = torch.clamp(torch.searchsorted(k_grid, log_strikes), min=1, max=Nfft - 1)
    k0 = k_grid[idx - 1]
    k1 = k_grid[idx]
    c0 = call_grid[idx - 1]
    c1 = call_grid[idx]
    weights_interp = (log_strikes - k0) / (k1 - k0)
    prices = c0 + (c1 - c0) * weights_interp
    return torch.clamp(prices, min=0.0)
