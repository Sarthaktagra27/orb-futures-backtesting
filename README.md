# ORB Futures Backtesting Engine

A quantitative backtesting framework for an **Opening Range Breakout (ORB)** strategy 
on NQ (Nasdaq-100) and DAX futures, built in Python using pandas and numpy.

## Overview

This project systematically analyses 6 years of 5-minute intraday futures data to 
identify the optimal combination of entry filters for an ORB strategy. The core 
research question was: **which measurable pre-market conditions predict whether an 
opening range breakout will follow through or fail?**

The analysis processed **420,000+ rows** of OHLC data across **1,857 trading days** 
and evaluated **10,000+ parameter combinations** via grid search.

---

## Key Findings

### 1. Previous-day range is the strongest predictor
The single most predictive variable was not price level or CPR alone, but the 
**previous session's trading range** — a volatility regime filter:

| Previous day range | NQ T1 Win Rate | DAX T1 Win Rate |
|---|---|---|
| < 150 pts | ~29% | ~30% |
| 150–225 pts | **42–52%** | ~33% |
| > 225 pts | ~35% | **37–39%** |

This means the two instruments perform best in **different volatility regimes** — 
a natural diversification that reduces correlation between them.

### 2. Cross-instrument negative correlation
NQ win rate drops from **42% to 27%** on days when DAX already made a clean 
directional move. Skipping NQ when DAX wins is validated across all 6 years.

### 3. Filter optimisation results

| Strategy | 6yr P&L | Win Rate | Max Drawdown | Profitable Years |
|---|---|---|---|---|
| No filters | -$15,800 | 28.2% | 71% | 0/6 |
| CPR + Range only | -$3,600 | 30.6% | 22% | 1/6 |
| + DAX conditional | -$1,200 | 32.2% | 17% | 2/6 |
| **+ Volatility regime** | **+$6,000** | **36.7%** | **12%** | **4/6** |

---

## Strategy Rules (Optimal Parameters)

### DAX ORB — 02:00 CT (Frankfurt 09:00 CET)
- Previous day range **> 225 pts** (high volatility regime)
- CPR width: **≤ 20 pts OR > 60 pts** (skip 20–60 choppy zone)
- Opening range: **60–100 pts**
- Skip FOMC and CPI days

### NQ ORB — 08:45 CT (NYSE 09:45 ET)
- Previous day range **150–225 pts** (moderate volatility sweet spot)
- DAX did **not win** today (conditional cross-instrument filter)
- NQ CPR width: **≤ 50 pts**
- Opening range: **60–130 pts**
- Skip FOMC and CPI days

---

## Project Structure

```
orb-futures-backtesting/
│
├── orb_backtest.py          # Main backtesting engine (well-commented)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # OHLC data loading and cleaning
│   ├── 02_cpr_analysis.ipynb        # CPR calculation and filter analysis
│   ├── 03_grid_search.ipynb         # Parameter optimisation
│   └── 04_results_analysis.ipynb    # Final results and visualisation
│
├── data/
│   └── README.md            # Data source instructions (Barchart)
│
└── README.md
```

---

## Technical Implementation

### Data pipeline
- Loads raw Barchart CSV exports (5-minute OHLC, multiple contract files)
- Handles futures contract rollovers via deduplication on timestamp
- Calculates session-level OHLC (Frankfurt session for DAX, US session for NQ)

### CPR calculation
```python
Pivot  = (High + Low + Close) / 3
BC     = (High + Low) / 2
TC     = (Pivot - BC) + Pivot
Width  = |TC - BC|
```

### ORB signal detection
- Opening range defined from first 3 candles after session open
- Breakout confirmed on close above/below range high/low
- Entry at next candle open, stop at range midpoint, target at 1:2 RR

### Grid search
- 10,000+ combinations of CPR thresholds, range bounds, and volatility filters
- Scored on composite metric: win rate, profit factor, positive years, max DD
- Walk-forward validated across individual years (not just aggregate)

---

## Data Sources

Data from [Barchart.com](https://www.barchart.com) — 5-minute intraday futures:
- **NQ**: Nasdaq-100 E-mini futures (continuous contract)
- **DAX**: DAX futures — DY contract (Barchart symbol: DY1!)

Data is not included in this repository. See `data/README.md` for download instructions.

---

## Dependencies

```bash
pip install pandas numpy openpyxl
```

---

## Results Summary (2020–2025, $200 risk per trade, 1:2 RR)

- **Total trades:** 294 (≈ 4/month average — selective, high-quality setups only)
- **Win rate:** 36.7% overall (T1: 38.6%, T2: 27.1%)
- **Profit factor:** 1.16
- **6-year P&L:** +$6,000
- **Max drawdown:** 12.0%
- **Profitable years:** 4/6

---

## Limitations and Caveats

- Results assume no slippage (real execution would reduce P&L slightly)
- Grid search was conducted on the full 6-year dataset — some overfitting risk
- DAX data only available from 2020; earlier NQ data trades without the DAX filter
- Strategy is selective (~4 trades/month) — small sample per year adds variance

---

*Built as a personal quantitative research project to develop and validate a 
systematic approach to short-term futures trading.*
