# Market-risk VaR engine

A Python study of portfolio market risk: daily P&L, Value at Risk (VaR), Expected Shortfall, rolling backtests and an illustrative FRTB P&L attribution test. The notebooks explain the calculations; reusable functions live in `src/`.

## Portfolio and data

The fixed-notional book holds equities, EUR and JPY exposures through ETFs, and a rates proxy:

| Instrument | Exposure | Signed USD notional |
|---|---|---:|
| AAPL | Apple equity | +1,000,000 |
| JPM | JPMorgan equity | +750,000 |
| SPY | S&P 500 ETF hedge | −2,000,000 |
| FXE | EUR ETF proxy | +1,500,000 |
| FXY | JPY ETF proxy | −1,000,000 |
| TLT | Long-duration Treasury ETF proxy | +1,250,000 |

The download requests daily adjusted Yahoo Finance prices for 2019–2024. The verified run loaded **1,509 aligned price days**, from **2019-01-02 to 2024-12-30**, giving **1,508 daily log-return observations**. Portfolio P&L is a linear approximation using signed notionals and those returns.

## Methods

- Historical VaR uses the empirical portfolio-loss quantile.
- Parametric VaR assumes normally distributed portfolio P&L.
- Monte Carlo VaR simulates multivariate normal instrument returns using their estimated means and covariance matrix, then aggregates position P&L.
- Historical Expected Shortfall averages losses beyond the VaR threshold.
- The rolling 99% backtest estimates each day's VaR from the preceding 250 observations and records exceptions.
- The traffic-light function scales the observed exception count to a 250-day equivalent. This is an approximation for the longer sample.
- Kupiec's proportion-of-failures test checks the exception frequency.
- The FRTB-style P&L attribution example compares hypothetical and risk-theoretical P&L using Spearman correlation and a two-sample Kolmogorov–Smirnov statistic. It includes deliberately incomplete risk-factor models and illustrative zone thresholds.

**The Christoffersen independence test is not implemented.** Exception clustering is therefore not formally tested here.

## Results from the real-data run

All five notebooks were executed in order. Each saved output confirms `real (Yahoo Finance)`. The [consolidated report](notebooks/day5_consolidated_report.ipynb) and [backtesting notebook](notebooks/day3_backtesting.ipynb) contain the results below.

| One-day risk measure at 99% confidence | USD, rounded |
|---|---:|
| Historical VaR | 42,438 |
| Parametric VaR | 41,017 |
| Monte Carlo VaR | 41,130 |
| Historical Expected Shortfall | 56,988 |

Historical VaR produced **22 exceptions over 1,258 backtest days**, versus **12.6 expected**. The observed exception rate was **1.749%**. The scaled traffic-light function returned **GREEN**, while the **Kupiec p-value was 0.016**, rejecting the target exception frequency at the usual 5% level. The green approximation does not establish adequate calibration; the frequency test flags underestimation of tail risk.

The attribution example's full-factor model was green; the deliberately incomplete model was amber on these real returns. This is a controlled model-comparison exercise, not regulatory approval.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install nbconvert ipykernel
```

Run the notebooks in filename order:

```bash
for notebook in notebooks/day*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$notebook" || break
done
```

Every notebook retains a synthetic-data fallback if Yahoo Finance is unavailable. Check its printed source marker after execution. If it reports `synthetic`, stop: those outputs must not be described as real market results. An execution exit code alone does not establish the data source.

For the reusable source modules:

```bash
python -m src.explore
python -m pytest -q
```

## Tests

The cleanup run passed **7 pytest tests**, and the consolidated notebook passed its inline checks. Tests cover analytic parametric VaR, tail-risk properties, simulated exception rates, Kupiec calibration and detection of missing attribution factors. The simulations in the tests are intentional controlled checks; the reported portfolio study above uses downloaded market data.

## Limitations

- A small fixed-notional linear book excludes options, nonlinear revaluation, funding costs, execution costs and liquidity risk.
- ETFs proxy FX and rates exposures; they are not spot currency or yield-curve factor models.
- Normal parametric and Monte Carlo models can miss heavy tails and changing dependence. The historical window can also miss a new regime.
- Traffic-light scaling and attribution thresholds are educational approximations, not a regulatory implementation or capital calculation.
- The Christoffersen independence test and broader conditional-coverage diagnostics are absent.
- Downloads, adjusted prices and library versions can change. Reruns should recheck the source markers and regenerate the results.
- Notebook calculation functions duplicate parts of `src/`; this cleanup changed saved execution outputs and documentation, without changing model logic.
