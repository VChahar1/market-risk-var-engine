"""Quick check: load data, build P&L, print basic stats."""

from src.portfolio import (
    download_prices,
    compute_returns,
    portfolio_pnl,
    signed_notionals,
)


def main():
    print("Downloading prices...")
    prices = download_prices()
    print(f"Loaded {prices.shape[0]} days x {prices.shape[1]} instruments")
    print(f"Range: {prices.index.min().date()} to {prices.index.max().date()}\n")

    returns = compute_returns(prices)
    w = signed_notionals()
    print("Signed USD exposure per instrument:")
    for t, val in w.items():
        print(f"  {t:>5}: {val:>+14,.0f}")
    print(f"  Gross: {w.abs().sum():>13,.0f}   Net: {w.sum():>+13,.0f}\n")

    pnl = portfolio_pnl(returns)
    print("Daily P&L (USD):")
    print(f"  Mean:            {pnl.mean():>+12,.0f}")
    print(f"  Std dev:         {pnl.std():>12,.0f}")
    print(f"  Worst day:       {pnl.min():>+12,.0f}")
    print(f"  Best day:        {pnl.max():>+12,.0f}")
    print(f"  5th percentile:  {pnl.quantile(0.05):>+12,.0f}")
    print(f"  1st percentile:  {pnl.quantile(0.01):>+12,.0f}")


if __name__ == "__main__":
    main()
