"Define a multi-asset trading book and fetch its market data."

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


PORTFOLIO = {
    "AAPL": (1_000_000, +1),
    "JPM": (750_000, +1),
    "SPY": (2_000_000, -1),
    "FXE": (1_500_000, +1),
    "FXY": (1_000_000, -1),
    "TLT": (1_250_000, +1),
}

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"


def download_prices() -> pd.DataFrame:
    """Download daily close prices for every instrument in the book."""
    tickers = list(PORTFOLIO.keys())
    raw = yf.download(
        tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False
    )["Close"]

    return raw[tickers].dropna()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns. Additive over time and symmetric — standard in risk."""
    return np.log(prices / prices.shift(1)).dropna()


def signed_notionals() -> pd.Series:
    """Signed USD exposure per instrument (notional * direction)."""
    return pd.Series(
        {t: notional * direction for t, (notional, direction) in PORTFOLIO.items()}
    )


def portfolio_pnl(returns: pd.DataFrame) -> pd.Series:
    "Daily P&L of the whole book in USD."
    w = signed_notionals()
    aligned = returns[w.index]
    return pd.Series(aligned.values @ w.values, index=aligned.index, name="pnl")
