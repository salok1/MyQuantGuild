"""
Helpers for turning CSV market data into PyTorch-friendly tensors.
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import torch

from bs_iv import iv_call_brent_torch

torch.set_default_dtype(torch.float64)


def make_dataset_from_csv(df: pd.DataFrame, r: float, q: float = 0.0) -> Dict[str, torch.Tensor]:
    """
    Build tensors for training the Heston parameter network.
    """
    S0 = torch.tensor(df["S0"].values, dtype=torch.float64)
    K = torch.tensor(df["K"].values, dtype=torch.float64)
    T = torch.tensor(df["T"].values, dtype=torch.float64)
    C_mkt = torch.tensor(df["C_mkt"].values, dtype=torch.float64)

    sigmas = []
    for s0, k, t, price in zip(S0.tolist(), K.tolist(), T.tolist(), C_mkt.tolist()):
        sigma = iv_call_brent_torch(price, s0, k, t, r, q)
        sigmas.append(float(sigma))
    sigma_bs = torch.tensor(sigmas, dtype=torch.float64)

    ratio = S0 / K
    X = torch.stack((ratio, T, sigma_bs), dim=1)

    return {
        "S0": S0,
        "K": K,
        "T": T,
        "r": torch.tensor(r, dtype=torch.float64),
        "X": X,
        "y": C_mkt,
    }


def split_train_val(data: Dict[str, torch.Tensor], val_frac: float = 0.2, seed: int = 123) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Deterministically shuffle and split tensors into train/validation sets.
    """
    n_samples = data["X"].shape[0]
    n_val = max(1, int(n_samples * val_frac))

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(n_samples, generator=generator)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    def subset(idx_tensor):
        return {
            key: value if key == "r" else value[idx_tensor]
            for key, value in data.items()
        }

    return subset(train_idx), subset(val_idx)
