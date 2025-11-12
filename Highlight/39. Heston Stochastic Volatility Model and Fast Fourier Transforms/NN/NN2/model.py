"""
Neural network that predicts Heston parameters and prices options.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from heston_torch import HestonParams, carr_madan_call_torch

torch.set_default_dtype(torch.float64)


class HestonParamNet(nn.Module):
    """Simple fully-connected network producing unconstrained parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, x: torch.Tensor) -> HestonParams:
        raw = self.net(x)
        return HestonParams.from_unconstrained(
            raw[..., 0],
            raw[..., 1],
            raw[..., 2],
            raw[..., 3],
            raw[..., 4],
        )


def price_with_params(
    S0: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    params: HestonParams,
    *,
    alpha: float = 1.5,
    Nfft: int = 2**12,
    eta: float = 0.25,
    q: float = 0.0,
) -> torch.Tensor:
    """
    Price each option using the Carr-Madan FFT evaluator.

    The current implementation iterates over samples for clarity, which keeps
    the code easy to follow and still differentiable.
    """
    prices = []
    for idx in range(S0.shape[0]):
        single_params = HestonParams(
            kappa=params.kappa[idx],
            theta=params.theta[idx],
            sigma=params.sigma[idx],
            rho=params.rho[idx],
            v0=params.v0[idx],
        )
        price = carr_madan_call_torch(
            S0[idx],
            r,
            q,
            T[idx],
            single_params,
            K[idx],
            alpha=alpha,
            Nfft=Nfft,
            eta=eta,
        )
        prices.append(price)
    return torch.stack(prices)


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root-mean-squared error with a small epsilon for stability."""
    mse = torch.mean((pred - target) ** 2)
    return torch.sqrt(mse + 1e-12)
