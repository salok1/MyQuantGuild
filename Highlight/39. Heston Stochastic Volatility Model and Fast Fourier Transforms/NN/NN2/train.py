"""
Training script for the Heston parameter neural network.
"""
from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd
import torch

from dataset import make_dataset_from_csv, split_train_val
from io_csv import read_market_csv
from model import HestonParamNet, price_with_params, rmse_loss

torch.set_default_dtype(torch.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Heston parameter NN.")
    parser.add_argument("--csv", required=True, help="Path to market CSV input.")
    parser.add_argument("--r", type=float, required=True, help="Risk-free rate.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--nfft", type=int, default=2**12)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default=None, help="Optional destination for model state dict.")
    return parser.parse_args()


def train_epoch(model, data, optimizer, alpha, nfft, eta):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    params = model(data["X"])
    preds = price_with_params(data["S0"], data["K"], data["T"], data["r"], params, alpha=alpha, Nfft=nfft, eta=eta)
    loss = rmse_loss(preds, data["y"])
    loss.backward()
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def evaluate(model, data, alpha, nfft, eta):
    model.eval()
    params = model(data["X"])
    preds = price_with_params(data["S0"], data["K"], data["T"], data["r"], params, alpha=alpha, Nfft=nfft, eta=eta)
    loss = rmse_loss(preds, data["y"])
    return loss, preds


def print_validation_table(S0, K, T, C_true, C_pred):
    rows = min(5, S0.shape[0])
    err = torch.abs(C_pred - C_true)
    rel_err = err / torch.clamp(torch.abs(C_true), min=1e-6) * 100.0
    table = pd.DataFrame(
        {
            "S0": S0[:rows].cpu().numpy(),
            "K": K[:rows].cpu().numpy(),
            "T": T[:rows].cpu().numpy(),
            "C_mkt": C_true[:rows].cpu().numpy(),
            "C_hat": C_pred[:rows].cpu().numpy(),
            "abs_err": err[:rows].cpu().numpy(),
            "rel_err_%": rel_err[:rows].cpu().numpy(),
        }
    )
    print(table.to_string(index=False, float_format=lambda x: f"{x: .6f}"))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    df = read_market_csv(args.csv)
    data = make_dataset_from_csv(df, r=args.r, q=0.0)
    train_data, val_data = split_train_val(data, val_frac=args.val_frac, seed=args.seed)

    model = HestonParamNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_rmse = train_epoch(model, train_data, optimizer, args.alpha, args.nfft, args.eta)
        val_rmse, val_preds = evaluate(model, val_data, args.alpha, args.nfft, args.eta)
        print(f"Epoch {epoch:03d} | Train RMSE: {train_rmse.item():.6f} | Val RMSE: {val_rmse.item():.6f}")
        print_validation_table(val_data["S0"], val_data["K"], val_data["T"], val_data["y"], val_preds)

    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model parameters saved to {args.save_path}")


if __name__ == "__main__":
    main()
