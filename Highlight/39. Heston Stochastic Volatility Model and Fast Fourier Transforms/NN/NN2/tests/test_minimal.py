"""
Minimal regression tests for the Heston NN codebase.
"""
from __future__ import annotations

import torch

from bs_iv import bs_call_torch, iv_call_brent_torch
from heston_torch import HestonParams, heston_cf
from model import HestonParamNet, price_with_params


def test_characteristic_function_normalization():
    params = HestonParams(
        kappa=torch.tensor(2.0),
        theta=torch.tensor(0.04),
        sigma=torch.tensor(0.5),
        rho=torch.tensor(-0.3),
        v0=torch.tensor(0.04),
    )
    value = heston_cf(torch.tensor(0.0, dtype=torch.complex128), 1.0, 100.0, 0.01, 0.0, params)
    assert torch.allclose(value, torch.tensor(1.0, dtype=torch.complex128), atol=1e-10)


def test_bs_price_iv_roundtrip():
    S0, K, T, r, q, vol = 100.0, 110.0, 0.7, 0.02, 0.0, 0.25
    price = bs_call_torch(S0, K, T, r, q, vol)
    recovered = iv_call_brent_torch(price, S0, K, T, r, q)
    assert torch.allclose(recovered, torch.tensor(vol), atol=1e-4)


def test_network_backward_pass():
    torch.manual_seed(0)
    batch = 3
    S0 = torch.full((batch,), 100.0)
    K = torch.linspace(90.0, 110.0, batch)
    T = torch.linspace(0.5, 1.0, batch)
    r = torch.tensor(0.01)

    X = torch.rand(batch, 3, requires_grad=True)
    model = HestonParamNet()
    params = model(X)
    prices = price_with_params(S0, K, T, r, params)
    loss = prices.sum()
    loss.backward()
    total_grad = sum(param.grad.abs().sum() for param in model.parameters() if param.grad is not None)
    assert total_grad > 0
