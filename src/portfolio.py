"""Define a multi-asset trading book and fetch its market data.

A realistic (if small) book spanning equities, FX, and rates — the kind of
mix a market-risk desk measures VaR across. Each position has a notional
(USD exposure) and a direction (+1 long, -1 short).

We use liquid, free instruments from Yahoo Finance so the whole study is
reproducible with no paid data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


# Each position: ticker -> (notional_usd, direction).  +1 long, -1 short.
PORTFOLIO = {
    # Equities
    "AAPL": (1_000_000, +1),    # long Apple
    "JPM":  (750_000,  +1),     # long JPMorgan
    "SPY":  (2_000_000, -1),    # short S&P 500 (hedge against the equity longs)
    # FX  (FXE = Euro ETF, FXY = Yen ETF — plain equity-style tickers, always load)
    "FXE":  (1_500_000, +1),    # long EUR
    "FXY":  (1_000_000, -1),    # short JPY
    # Rates proxy
    "TLT":  (1_250_000, +1),    # long long-dated US Treasuries
}

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"


def download_prices() -> pd.DataFrame:
    """Download daily close prices for every instrument in the book."""
    tickers = list(PORTFOLIO.keys())
    raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)["Close"]
    # Align dates across markets; drop any day missing a price.
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
    """Daily P&L of the whole book in USD.

    For each day: sum over positions of signed_notional * instrument_return.
    A position of $N in an asset that moves by return r makes N*r. This linear
    approximation is exact enough for a daily-horizon book of linear assets.
    """
    w = signed_notionals()
    aligned = returns[w.index]
    return pd.Series(aligned.values @ w.values, index=aligned.index, name="pnl")