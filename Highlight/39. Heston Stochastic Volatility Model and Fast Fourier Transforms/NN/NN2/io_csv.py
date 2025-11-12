"""
Utility helpers for loading market data used by the Heston NN experiments.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd


REQUIRED_COLUMNS: Sequence[str] = ("S0", "K", "C_mkt", "T")


def read_market_csv(path: str) -> pd.DataFrame:
    """
    Read a CSV file containing option quotes.

    The file must contain the columns S0, K, C_mkt, and T.  Validation ensures
    that later stages of the pipeline can rely on these headers being present
    and correctly spelled.

    Parameters
    ----------
    path:
        Filesystem path to a CSV file readable by pandas.

    Returns
    -------
    pandas.DataFrame
        Data frame restricted to the required columns.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df[list(REQUIRED_COLUMNS)].copy()
