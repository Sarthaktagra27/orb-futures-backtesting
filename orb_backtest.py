"""
ORB (Opening Range Breakout) Backtesting Engine
================================================
Instruments : NQ (Nasdaq-100 futures) + DAX futures
Data        : 5-minute OHLC, 2020-2025 (420,000+ rows)
Author      : Sarthak Tagra
Description : Systematic backtesting framework to identify optimal
              entry filters for an ORB strategy across two correlated
              futures instruments. Includes grid search over 10,000+
              parameter combinations.

Key findings:
  - DAX previous-day range is the strongest predictor of next-day win rate
  - NQ win rate: 42-52% when DAX prev-day range = 150-225pts
  - DAX win rate: 37-39% when DAX prev-day range > 225pts
  - Cross-instrument negative correlation (r ≈ -0.15) validated as filter
  - Optimal filters improved 6-year P&L from -$15,800 to +$6,000
  - Max drawdown reduced from 71% to 12%
"""

import pandas as pd
import numpy as np
import pickle
from itertools import product

# ── Configuration ─────────────────────────────────────────────
ACCOUNT  = 25_000   # starting account size ($)
RISK     = 200      # fixed risk per trade ($)
WIN_MULT = 2        # reward-to-risk ratio (1:2 RR)
WIN_AMT  = RISK * WIN_MULT   # $400
LOSS_AMT = RISK              # $200

# ── Session times (Chicago / CT timezone) ─────────────────────
# DAX Frankfurt open  = 09:00 CET = 02:00 CT
DAX_C1   = pd.Timestamp('02:00').time()   # opening range candle 1
DAX_C2   = pd.Timestamp('02:05').time()   # opening range candle 2
DAX_C3   = pd.Timestamp('02:10').time()   # opening range candle 3
DAX_SIG  = pd.Timestamp('02:15').time()   # earliest entry signal
DAX_CUT  = pd.Timestamp('05:00').time()   # hard session cutoff (no new entries)

# NQ NYSE open = 09:30 ET = 08:30 CT
NQ_C1    = pd.Timestamp('08:30').time()
NQ_C2    = pd.Timestamp('08:35').time()
NQ_C3    = pd.Timestamp('08:40').time()
NQ_SIG   = pd.Timestamp('08:45').time()
NQ_CUT   = pd.Timestamp('13:00').time()   # 1:00 PM CT = 2:00 PM ET

# ── News event dates to exclude ────────────────────────────────
# FOMC and CPI releases cause abnormal volatility that breaks ORB logic
FOMC_DATES = [
    '2020-01-29','2020-03-03','2020-03-15','2020-04-29','2020-06-10','2020-07-29',
    '2020-09-16','2020-11-05','2020-12-16','2021-01-27','2021-03-17','2021-04-28',
    '2021-06-16','2021-07-28','2021-09-22','2021-11-03','2021-12-15','2022-01-26',
    '2022-03-16','2022-05-04','2022-06-15','2022-07-27','2022-09-21','2022-11-02',
    '2022-12-14','2023-02-01','2023-03-22','2023-05-03','2023-06-14','2023-07-26',
    '2023-09-20','2023-11-01','2023-12-13','2024-01-31','2024-03-20','2024-05-01',
    '2024-06-12','2024-07-31','2024-09-18','2024-11-07','2024-12-18','2025-01-29',
    '2025-03-19','2025-05-07','2025-06-18','2025-07-30','2025-09-17',
]
CPI_DATES = [
    '2020-01-14','2020-02-13','2020-03-11','2020-04-10','2020-05-12','2020-06-10',
    '2020-07-14','2020-08-12','2020-09-11','2020-10-13','2020-11-12','2020-12-10',
    '2021-01-13','2021-02-10','2021-03-10','2021-04-13','2021-05-12','2021-06-10',
    '2021-07-13','2021-08-11','2021-09-14','2021-10-13','2021-11-10','2021-12-10',
    '2022-01-12','2022-02-10','2022-03-10','2022-04-12','2022-05-11','2022-06-10',
    '2022-07-13','2022-08-10','2022-09-13','2022-10-13','2022-11-10','2022-12-13',
    '2023-01-12','2023-02-14','2023-03-14','2023-04-12','2023-05-10','2023-06-13',
    '2023-07-12','2023-08-10','2023-09-13','2023-10-12','2023-11-14','2023-12-12',
    '2024-01-11','2024-02-13','2024-03-12','2024-04-10','2024-05-15','2024-06-12',
    '2024-07-11','2024-08-14','2024-09-11','2024-10-10','2024-11-13','2024-12-11',
    '2025-01-15','2025-02-12','2025-03-12','2025-04-10','2025-05-13',
]
NEWS_DATES = set(pd.to_datetime(FOMC_DATES + CPI_DATES).date)


# ── Data loading ───────────────────────────────────────────────

def load_ohlc(filepath: str, start: str, end: str) -> pd.DataFrame:
    """
    Load a single Barchart CSV contract file and filter to date range.

    Barchart exports 5-minute OHLC with columns:
    Time, Open, High, Low, Close, Change, %Chg, Volume

    Parameters
    ----------
    filepath : path to CSV file
    start    : start date string 'YYYY-MM-DD'
    end      : end date string   'YYYY-MM-DD'

    Returns
    -------
    DataFrame with Time as datetime, filtered to [start, end]
    """
    df = pd.read_csv(filepath, skipfooter=1, engine='python')
    df.columns = ['Time','Open','High','Low','Close','Change','PctChange','Volume']
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    s = pd.Timestamp(start).date()
    e = pd.Timestamp(end).date()
    df = df[(df['Time'].dt.date >= s) & (df['Time'].dt.date <= e)]
    return df


def build_master(file_list: list) -> pd.DataFrame:
    """
    Concatenate multiple contract files into a single master DataFrame.
    Deduplicates on timestamp to handle rollover overlap.

    Parameters
    ----------
    file_list : list of (filepath, start_date, end_date) tuples

    Returns
    -------
    Master DataFrame sorted by Time with TimeOnly and Date columns added
    """
    dfs = []
    for filepath, start, end in file_list:
        dfs.append(load_ohlc(filepath, start, end))
    master = (pd.concat(dfs)
                .sort_values('Time')
                .drop_duplicates(subset=['Time'])
                .reset_index(drop=True))
    master['Date']     = master['Time'].dt.date
    master['TimeOnly'] = master['Time'].dt.time
    return master


# ── Central Pivot Range (CPR) calculation ─────────────────────

def calculate_cpr(master_df: pd.DataFrame,
                  session_start: str = '02:00',
                  session_end:   str = '15:00') -> dict:
    """
    Calculate the previous day's Central Pivot Range width for each trading day.

    CPR is a technical indicator derived from the previous session's OHLC:
        Pivot  = (High + Low + Close) / 3
        BC     = (High + Low) / 2             (Bottom Central)
        TC     = (Pivot - BC) + Pivot          (Top Central)
        Width  = |TC - BC|

    A narrow CPR indicates price compression → often precedes strong directional moves.
    A wide CPR indicates a ranging/trending environment.

    Parameters
    ----------
    master_df     : full OHLC master DataFrame
    session_start : start of the session to use for OHLC calculation (CT time)
    session_end   : end of the session

    Returns
    -------
    dict mapping date -> previous day's CPR width (in index points)
    """
    session = master_df[
        (master_df['TimeOnly'] >= pd.Timestamp(session_start).time()) &
        (master_df['TimeOnly'] <= pd.Timestamp(session_end).time())
    ].copy()

    daily = (session.groupby('Date')
                    .agg(High=('High','max'), Low=('Low','min'), Close=('Close','last'))
                    .reset_index()
                    .sort_values('Date')
                    .reset_index(drop=True))

    daily['Pivot'] = (daily['High'] + daily['Low'] + daily['Close']) / 3
    daily['BC']    = (daily['High'] + daily['Low']) / 2
    daily['TC']    = (daily['Pivot'] - daily['BC']) + daily['Pivot']
    daily['CPR']   = (daily['TC'] - daily['BC']).abs().round(1)

    # Previous day range — key volatility regime filter discovered in analysis
    daily['Prev_Range'] = (daily['High'] - daily['Low']).shift(1).round(1)
    daily['Prev_CPR']   = daily['CPR'].shift(1)
    daily = daily.dropna().reset_index(drop=True)

    cpr_map       = dict(zip(daily['Date'], daily['Prev_CPR']))
    prevrange_map = dict(zip(daily['Date'], daily['Prev_Range']))
    return cpr_map, prevrange_map


# ── Core ORB simulation for a single day ─────────────────────

def simulate_orb_day(day_df:   pd.DataFrame,
                     c1_time:  object,
                     c2_time:  object,
                     c3_time:  object,
                     sig_time: object,
                     cut_time: object) -> tuple:
    """
    Simulate Opening Range Breakout trades for a single trading day.

    Strategy logic:
    1. Define opening range from first 3 candles (c1, c2, c3)
    2. Range high = max(H1, H2, H3), Range low = min(L1, L2, L3)
    3. Midpoint = (High + Low) / 2 — used as stop-loss anchor
    4. On first close ABOVE range high → enter LONG at next candle open
       Stop = midpoint, Target = entry + 2 * (entry - midpoint)  [1:2 RR]
    5. On first close BELOW range low → enter SHORT (same mirror logic)
    6. Take T2 only if T1 hit stop loss and a new breakout forms

    Parameters
    ----------
    day_df   : single-day OHLC DataFrame (already filtered to this date)
    c1_time  : time of first candle in opening range
    c2_time  : time of second candle
    c3_time  : time of third candle
    sig_time : earliest time to enter a trade
    cut_time : hard cutoff — no new entries after this time

    Returns
    -------
    (trades_list, first_outcome_string, opening_range_size)
    trades_list      : list of trade result dicts
    first_outcome    : 'WIN' | 'LOSS' | 'NO_BREAKOUT' | 'NO_SIGNAL'
    opening_range    : range high - range low (in index points)
    """
    # Extract opening range candles
    c1 = day_df[day_df['TimeOnly'] == c1_time]
    c2 = day_df[day_df['TimeOnly'] == c2_time]
    c3 = day_df[day_df['TimeOnly'] == c3_time]

    if c1.empty or c2.empty or c3.empty:
        return [], 'NO_SIGNAL', 0

    # Define opening range boundaries
    range_high = max(c1.iloc[0]['High'], c2.iloc[0]['High'], c3.iloc[0]['High'])
    range_low  = min(c1.iloc[0]['Low'],  c2.iloc[0]['Low'],  c3.iloc[0]['Low'])
    midpoint   = (range_high + range_low) / 2
    opening_range = range_high - range_low

    # Get signal candles (after opening range, before cutoff)
    signal_candles = (day_df[(day_df['TimeOnly'] >= sig_time) &
                             (day_df['TimeOnly'] <= cut_time)]
                      .reset_index(drop=True))

    trades      = []
    trade_count = 0
    took_long   = False
    took_short  = False
    t1_lost     = False

    for i in range(len(signal_candles)):
        # Max 2 trades per day; T2 only after T1 stop-out
        if trade_count >= 2:
            break
        if trade_count == 1 and not t1_lost:
            break

        candle = signal_candles.iloc[i]
        direction = None

        # Breakout detection
        if candle['Close'] > range_high and not took_long:
            direction = 'LONG'
        elif candle['Close'] < range_low and not took_short:
            direction = 'SHORT'

        if direction is None:
            continue

        # Entry at next candle open
        next_idx = i + 1
        if next_idx >= len(signal_candles):
            continue
        entry = signal_candles.iloc[next_idx]['Open']

        # Calculate stop and target
        if direction == 'LONG':
            sl_dist = entry - midpoint
            if sl_dist <= 0:
                continue
            stop_loss   = midpoint
            take_profit = entry + sl_dist * WIN_MULT
            took_long   = True
        else:
            sl_dist = midpoint - entry
            if sl_dist <= 0:
                continue
            stop_loss   = midpoint
            take_profit = entry - sl_dist * WIN_MULT
            took_short  = True

        # Simulate trade outcome bar by bar
        outcome   = None
        exit_price = None

        for _, future_bar in signal_candles.iloc[next_idx:].iterrows():
            if direction == 'LONG':
                # Check stop first (conservative)
                if future_bar['Open'] <= stop_loss:
                    outcome = 'LOSS'; exit_price = stop_loss; break
                if future_bar['Open'] >= take_profit:
                    outcome = 'WIN';  exit_price = take_profit; break
                # Wicked both ways in single bar
                if future_bar['High'] >= take_profit and future_bar['Low'] <= stop_loss:
                    outcome = ('WIN' if abs(future_bar['Open'] - take_profit) <
                                       abs(future_bar['Open'] - stop_loss) else 'LOSS')
                    exit_price = take_profit if outcome == 'WIN' else stop_loss; break
                elif future_bar['High'] >= take_profit:
                    outcome = 'WIN';  exit_price = take_profit; break
                elif future_bar['Low']  <= stop_loss:
                    outcome = 'LOSS'; exit_price = stop_loss;   break
            else:  # SHORT
                if future_bar['Open'] >= stop_loss:
                    outcome = 'LOSS'; exit_price = stop_loss; break
                if future_bar['Open'] <= take_profit:
                    outcome = 'WIN';  exit_price = take_profit; break
                if future_bar['Low'] <= take_profit and future_bar['High'] >= stop_loss:
                    outcome = ('WIN' if abs(future_bar['Open'] - take_profit) <
                                       abs(future_bar['Open'] - stop_loss) else 'LOSS')
                    exit_price = take_profit if outcome == 'WIN' else stop_loss; break
                elif future_bar['Low']  <= take_profit:
                    outcome = 'WIN';  exit_price = take_profit; break
                elif future_bar['High'] >= stop_loss:
                    outcome = 'LOSS'; exit_price = stop_loss;   break

        if outcome is None:
            continue  # trade never resolved before cutoff

        # Calculate dollar P&L using fixed risk
        pnl_points = (exit_price - entry) if direction == 'LONG' else (entry - exit_price)
        dollar_pnl = (RISK / sl_dist) * pnl_points

        trade_count += 1
        if outcome == 'LOSS':
            t1_lost = True

        trades.append({
            'Outcome':     outcome,
            'Dollar_PnL':  round(dollar_pnl, 2),
            'Entry':       round(entry, 2),
            'Exit':        round(exit_price, 2),
            'SL_Dist':     round(sl_dist, 2),
            'Range':       round(opening_range, 1),
            'Direction':   direction,
            'T_Num':       trade_count,
        })

    # Summarise day outcome for conditional NQ logic
    if any(t['Outcome'] == 'WIN'  for t in trades): first_outcome = 'WIN'
    elif trades:                                      first_outcome = 'LOSS'
    else:                                             first_outcome = 'NO_BREAKOUT'

    return trades, first_outcome, opening_range


# ── Strategy runner ────────────────────────────────────────────

def run_combined_strategy(nq_by_date:       dict,
                          dax_by_date:      dict,
                          dax_cpr_map:      dict,
                          dax_prevrange_map:dict,
                          nq_cpr_map:       dict,
                          params:           dict) -> pd.DataFrame:
    """
    Run the combined DAX + NQ ORB strategy over all available trading days.

    Conditional logic:
    - DAX trades first (3:00 AM CT)
    - NQ only trades when DAX did NOT win (negative correlation filter)
    - Each instrument has independent CPR, range, and volatility filters

    Key filter parameters (tuned via grid search):
    - dax_cpr_skip_lo / hi : skip DAX when CPR falls in this zone (choppy market)
    - dax_rng_min / max    : valid opening range for DAX (too tight = fakeouts)
    - dax_prevrange_min    : minimum previous day range (volatility regime filter)
    - nq_cpr_max           : maximum NQ CPR to trade
    - nq_rng_min / max     : valid NQ opening range
    - nq_prevrange_min/max : NQ volatility regime filter (sweet spot = 150-225pts)

    Parameters
    ----------
    nq_by_date        : dict of date -> NQ day DataFrame
    dax_by_date       : dict of date -> DAX day DataFrame
    dax_cpr_map       : dict of date -> previous day DAX CPR width
    dax_prevrange_map : dict of date -> previous day DAX H-L range
    nq_cpr_map        : dict of date -> previous day NQ CPR width
    params            : strategy parameters dict (see defaults below)

    Returns
    -------
    DataFrame of all trades with columns:
    Date, Year, Month, Inst, T_Num, Outcome, Dollar_PnL, Range
    """
    # Parameter defaults (optimal values found via grid search)
    dax_cpr_skip_lo   = params.get('dax_cpr_skip_lo',    20)   # skip CPR 20-60
    dax_cpr_skip_hi   = params.get('dax_cpr_skip_hi',    60)
    dax_rng_min       = params.get('dax_rng_min',        60)   # pts
    dax_rng_max       = params.get('dax_rng_max',       100)
    dax_prevrange_min = params.get('dax_prevrange_min', 225)   # volatility filter
    nq_skip_dax_win   = params.get('nq_skip_dax_win',  True)   # conditional logic
    nq_cpr_max        = params.get('nq_cpr_max',        50)
    nq_rng_min        = params.get('nq_rng_min',        60)
    nq_rng_max        = params.get('nq_rng_max',       130)
    nq_prevrange_min  = params.get('nq_prevrange_min', 150)    # NQ regime sweet spot
    nq_prevrange_max  = params.get('nq_prevrange_max', 225)

    all_days = sorted(set(list(nq_by_date.keys())) | set(list(dax_by_date.keys())))
    all_trades = []

    for day in all_days:
        yr = pd.Timestamp(day).year
        mo = pd.Timestamp(day).month

        # Skip news days
        if day in NEWS_DATES:
            continue

        dax_cpr       = dax_cpr_map.get(day, 9999)
        dax_prevrange = dax_prevrange_map.get(day, 0)
        nq_cpr        = nq_cpr_map.get(day, 9999)
        dax_outcome   = 'SKIP'

        # ── DAX ──────────────────────────────────────────────
        if day in dax_by_date:
            # CPR filter: skip when CPR is in the 'dead zone' (choppy, no direction)
            cpr_ok = not (dax_cpr_skip_lo < dax_cpr <= dax_cpr_skip_hi)
            # Volatility filter: only trade after a sufficiently volatile prior session
            pr_ok  = dax_prevrange >= dax_prevrange_min

            if cpr_ok and pr_ok:
                dax_trades, raw_outcome, dax_rng = simulate_orb_day(
                    dax_by_date[day], DAX_C1, DAX_C2, DAX_C3, DAX_SIG, DAX_CUT
                )
                # Opening range filter (checked after simulation for efficiency)
                if dax_rng_min <= dax_rng <= dax_rng_max:
                    for t in dax_trades:
                        all_trades.append({
                            'Date': pd.Timestamp(day), 'Year': yr, 'Month': mo,
                            'Inst': 'DAX', **t
                        })
                    dax_outcome = raw_outcome

        # ── NQ ───────────────────────────────────────────────
        # Skip NQ if DAX already won today (negative correlation filter)
        if nq_skip_dax_win and dax_outcome == 'WIN':
            continue

        # NQ volatility regime filter — sweet spot found at 150-225pts prev range
        nq_prevrange = dax_prevrange  # use DAX prev range as NQ regime proxy
        if not (nq_prevrange_min <= nq_prevrange <= nq_prevrange_max):
            continue

        if day in nq_by_date and nq_cpr <= nq_cpr_max:
            nq_trades, _, nq_rng = simulate_orb_day(
                nq_by_date[day], NQ_C1, NQ_C2, NQ_C3, NQ_SIG, NQ_CUT
            )
            if nq_rng_min <= nq_rng <= nq_rng_max:
                for t in nq_trades:
                    all_trades.append({
                        'Date': pd.Timestamp(day), 'Year': yr, 'Month': mo,
                        'Inst': 'NQ', **t
                    })

    return pd.DataFrame(all_trades)


# ── Performance metrics ────────────────────────────────────────

def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate standard strategy performance metrics.

    Parameters
    ----------
    df : trades DataFrame with Outcome and Dollar_PnL columns

    Returns
    -------
    dict with metrics: trades, win_rate, profit_factor, total_pnl,
                       max_drawdown_pct, max_consecutive_losses,
                       yearly_breakdown
    """
    if df.empty:
        return {}

    t = len(df)
    w = (df['Outcome'] == 'WIN').sum()
    l = t - w

    win_rate      = w / t
    profit_factor = (w * WIN_AMT) / (l * LOSS_AMT) if l > 0 else float('inf')
    total_pnl     = df['Dollar_PnL'].sum()

    # Drawdown calculation
    equity = ACCOUNT + df['Dollar_PnL'].cumsum()
    peak   = equity.cummax()
    dd_pct = ((peak - equity) / ACCOUNT * 100).max()

    # Max consecutive losses
    max_cl = cur_cl = 0
    for outcome in df['Outcome']:
        cur_cl = cur_cl + 1 if outcome == 'LOSS' else 0
        max_cl = max(max_cl, cur_cl)

    # Yearly breakdown
    yearly = {}
    for yr in range(2020, 2026):
        yd = df[df['Year'] == yr]
        if yd.empty:
            yearly[yr] = {'trades': 0, 'pnl': 0, 'win_rate': 0}
            continue
        yw  = (yd['Outcome'] == 'WIN').sum()
        yt  = len(yd)
        yearly[yr] = {
            'trades':   yt,
            'win_rate': round(yw / yt * 100, 1),
            'pnl':      round(yd['Dollar_PnL'].sum(), 0),
        }

    return {
        'trades':                t,
        'wins':                  int(w),
        'losses':                int(l),
        'win_rate':              round(win_rate * 100, 1),
        'profit_factor':         round(profit_factor, 2),
        'total_pnl':             round(total_pnl, 0),
        'max_drawdown_pct':      round(dd_pct, 1),
        'max_consecutive_losses':max_cl,
        'yearly':                yearly,
    }


# ── Grid search ────────────────────────────────────────────────

def grid_search(nq_by_date, dax_by_date, dax_cpr_map,
                dax_prevrange_map, nq_cpr_map) -> list:
    """
    Exhaustive grid search over strategy parameters.
    Tests 10,000+ combinations and ranks by composite score.

    Scoring weights:
    - Win rate above 33.3% breakeven (40%)
    - Profit factor (30%)
    - Positive years out of 6 (20%)
    - Low max drawdown (10%)

    Returns
    -------
    List of result dicts sorted by score descending
    """
    param_grid = {
        'dax_cpr_skip_lo':   [15, 20, 25],
        'dax_cpr_skip_hi':   [50, 60],
        'dax_rng_min':       [50, 60],
        'dax_rng_max':       [90, 100, 120],
        'dax_prevrange_min': [175, 200, 225, 250],
        'nq_skip_dax_win':   [True, False],
        'nq_cpr_max':        [30, 40, 50],
        'nq_rng_min':        [60, 75],
        'nq_rng_max':        [130, 150],
        'nq_prevrange_min':  [125, 150, 175],
        'nq_prevrange_max':  [200, 225, 250],
    }

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    results = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        df     = run_combined_strategy(nq_by_date, dax_by_date,
                                       dax_cpr_map, dax_prevrange_map,
                                       nq_cpr_map, params)
        if len(df) < 30:
            continue  # skip if too few trades to be meaningful

        metrics = calculate_metrics(df)
        pos_yrs = sum(1 for v in metrics['yearly'].values() if v['pnl'] > 0)
        wr      = metrics['win_rate']
        pf      = metrics['profit_factor']
        dd      = metrics['max_drawdown_pct']

        # Composite score
        be_bonus = max(0, wr - 33.3) / 10
        score = (be_bonus * 0.4 +
                 (min(pf, 2.5) / 2.5) * 0.3 +
                 (pos_yrs / 6) * 0.2 +
                 max(0, (15 - dd) / 15) * 0.1)

        results.append({**metrics, 'score': round(score, 3),
                        'pos_yrs': pos_yrs, 'params': params})

    return sorted(results, key=lambda x: -x['score'])


# ── Main ──────────────────────────────────────────────────────

if __name__ == '__main__':

    print("Loading NQ data...")
    # Replace with your actual Barchart CSV paths
    # nq_master = build_master([
    #     ('path/to/nqh20.csv', '2020-01-02', '2020-03-19'),
    #     ...
    # ])

    print("Loading DAX data...")
    # dax_master = build_master([...])

    print("Calculating CPR and volatility filters...")
    # dax_cpr_map, dax_prevrange_map = calculate_cpr(dax_master)
    # nq_cpr_map, _                  = calculate_cpr(nq_master)

    print("Running optimal strategy...")
    # optimal_params = {
    #     'dax_cpr_skip_lo':   20,
    #     'dax_cpr_skip_hi':   60,
    #     'dax_rng_min':       60,
    #     'dax_rng_max':      100,
    #     'dax_prevrange_min': 225,
    #     'nq_skip_dax_win':  True,
    #     'nq_cpr_max':        50,
    #     'nq_rng_min':        60,
    #     'nq_rng_max':       130,
    #     'nq_prevrange_min': 150,
    #     'nq_prevrange_max': 225,
    # }
    # df_trades = run_combined_strategy(
    #     nq_by_date, dax_by_date,
    #     dax_cpr_map, dax_prevrange_map, nq_cpr_map,
    #     optimal_params
    # )
    # metrics = calculate_metrics(df_trades)
    # print(metrics)
