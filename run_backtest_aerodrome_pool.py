#!/usr/bin/env python3
"""Delta-neutral backtest for a real Aerodrome pool using CSV inputs.

This script:
  - Loads ETH 1-minute prices and resamples to 15-minute bars
  - Loads daily pool volume from Snowflake CSV and converts it into per-step fees
  - Assumes a constant TVL (e.g. $1M) for this first iteration
  - Runs a delta-neutral LP strategy:
        * LP in a ±10% range around price
        * Perp short hedge (delta-neutral, funding ignored)
        * Swap fees from real volume
  - Reuses the existing LP engine and plotting style from the original backtest
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Ensure we can import project modules when running this file directly
import sys
ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DESKTOP_DIR = Path.home() / "Desktop"

from lp_engine import (  # type: ignore
    LPPosition,
    PerpHedge,
    calculate_arbitrage_loss,
    DeltaNeutralResult,
    DEFAULT_PERP_TRADING_COST,
)
from delta_neutral_backtest import plot_results  # type: ignore


# === CONFIG ====================================================================

DATA_DIR = ROOT / "data"
# Your Snowflake export currently lives under data/snowflake
SNOWFLAKE_DIR = DATA_DIR / "snowflake"

ETH_CSV = DATA_DIR / "eth_1min.csv"
# Adjust this if you rename/move the volume CSV
VOLUME_CSV = SNOWFLAKE_DIR / "Volume_05:01:2024-10:30:2025.csv"

START_DATE = pd.Timestamp("2024-05-01")
END_DATE = pd.Timestamp("2025-01-31")

RANGE_WIDTH_PCT = 0.10          # ±10% around mid price
REBALANCE_MINUTES = 15          # rebalance every 15 minutes (same as step)
FEE_RATE = 0.000356             # 0.0356% Aerodrome fee tier
TRADING_COST_PER_NOTIONAL = 0.0001  # 1 bp per hedge rebalance (adjust as desired)
INITIAL_TVL = 1_000_000.0       # Constant TVL for this iteration


# === DATA LOADING & PREP =======================================================

def load_eth_price_15m(
    csv_path: Path = ETH_CSV,
    start: pd.Timestamp = START_DATE,
    end: pd.Timestamp = END_DATE,
) -> pd.Series:
    """Load 1-min ETH price CSV and resample to 15-minute close."""
    if not csv_path.exists():
        raise FileNotFoundError(f"ETH price CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    # Try to infer timestamp and price columns
    ts_col = None
    for c in df.columns:
        if c.lower() in ("timestamp", "date", "time"):
            ts_col = c
            break
    if ts_col is None:
        raise ValueError(f"Could not find timestamp column in {csv_path}, got {list(df.columns)}")

    price_col = None
    for c in df.columns:
        if c.lower() in ("price", "close", "ethusdt", "eth_price"):
            price_col = c
            break
    if price_col is None:
        # Fallback: use the second column as price
        price_col = df.columns[1]

    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()

    df = df.loc[(df.index >= start) & (df.index <= end)]
    if df.empty:
        raise ValueError("ETH price data is empty after filtering to backtest window.")

    price_15m = df[price_col].resample("15min").last().ffill()
    price_15m.name = "price"
    return price_15m


def load_daily_volume_and_fees(
    price_15m: pd.Series,
    csv_path: Path = VOLUME_CSV,
    fee_rate: float = FEE_RATE,
) -> Tuple[pd.Series, pd.Series]:
    """Load daily volume from Snowflake CSV and compute:

    - daily_usd_volume: USD notional per day
    - daily_fee_usd: fee revenue per day
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Volume CSV not found at {csv_path}")

    df_vol = pd.read_csv(csv_path)
    lower_map = {c.lower(): c for c in df_vol.columns}

    day_col = lower_map.get("day")
    total_vol_col = lower_map.get("total_raw_volume")
    if day_col is None or total_vol_col is None:
        raise ValueError(
            "Volume CSV must contain DAY / TOTAL_RAW_VOLUME (any case). "
            f"Got columns: {list(df_vol.columns)}"
        )

    df_vol[day_col] = pd.to_datetime(df_vol[day_col]).dt.normalize()
    df_vol = df_vol.sort_values(day_col)

    # Align to backtest window
    df_vol = df_vol.loc[
        (df_vol[day_col] >= START_DATE.normalize())
        & (df_vol[day_col] <= END_DATE.normalize())
    ]
    if df_vol.empty:
        raise ValueError("Volume data is empty after filtering to backtest window.")

    vol_raw = df_vol[total_vol_col].astype(float)

    # Normalize on-chain raw volume (18 decimals) to token units
    if float(vol_raw.median()) > 1e15:
        vol_token_units = vol_raw / 1e18
    else:
        vol_token_units = vol_raw

    # Daily ETH price from 15m series
    daily_eth = price_15m.resample("1D").mean()

    # Align daily_eth to volume days
    daily_eth = daily_eth.reindex(df_vol[day_col].unique()).ffill()

    daily_usd_volume = vol_token_units.values * daily_eth.values
    daily_usd_volume = pd.Series(
        daily_usd_volume,
        index=pd.to_datetime(df_vol[day_col].values),
        name="daily_usd_volume",
    )

    daily_fee_usd = daily_usd_volume * fee_rate
    daily_fee_usd.name = "daily_fee_usd"

    return daily_usd_volume, daily_fee_usd


def build_fees_per_step(
    price_15m: pd.Series,
    daily_fee_usd: pd.Series,
) -> pd.Series:
    """Spread daily fees evenly across all 15-minute steps in that day."""
    idx = price_15m.index
    fees_per_step = pd.Series(0.0, index=idx, name="fees_per_step")

    # Normalize indices to dates
    step_days = idx.normalize()
    fee_days = daily_fee_usd.index.normalize()

    for day in fee_days:
        fee = float(daily_fee_usd.loc[day])
        mask = step_days == day
        n_steps = int(mask.sum())
        if n_steps == 0:
            continue
        fees_per_step.loc[mask] = fee / n_steps

    return fees_per_step


# === CORE BACKTEST =============================================================

def _lp_unit_value(lower: float, upper: float, price: float) -> float:
    """Value (in token1 units) of a Uniswap v3 LP position with L = 1."""
    sqrtP = np.sqrt(price)
    sqrtPa = np.sqrt(lower)
    sqrtPb = np.sqrt(upper)

    if price <= lower:
        amount0 = (sqrtPb - sqrtPa) / (sqrtPa * sqrtPb)
        amount1 = 0.0
    elif price >= upper:
        amount0 = 0.0
        amount1 = (sqrtPb - sqrtPa)
    else:
        amount0 = (sqrtPb - sqrtP) / (sqrtP * sqrtPb)
        amount1 = (sqrtP - sqrtPa)
    return amount0 * price + amount1


def run_backtest(
    price_15m: pd.Series,
    fees_per_step: pd.Series,
    range_width: float = RANGE_WIDTH_PCT,
    rebalance_minutes: int = REBALANCE_MINUTES,
    initial_tvl: float = INITIAL_TVL,
    trading_cost_per_notional: float = TRADING_COST_PER_NOTIONAL,
) -> DeltaNeutralResult:
    """Run a delta-neutral LP backtest on 15-minute data."""
    index = price_15m.index
    prices = price_15m.values.astype(float)

    P0 = float(prices[0])
    k = 1.0 + range_width
    lower0 = P0 / k
    upper0 = P0 * k

    unit_val0 = _lp_unit_value(lower0, upper0, P0)
    if unit_val0 <= 0:
        raise ValueError("Initial LP unit value is non-positive; check price/range data.")
    initial_liquidity = initial_tvl / unit_val0

    lp = LPPosition(lower=lower0, upper=upper0, liquidity=float(initial_liquidity), price=P0)
    hedge = PerpHedge(size=0.0, entry_price=P0, current_price=P0)

    total_fees = 0.0
    total_arb_loss = 0.0
    total_trading_costs = 0.0
    hedge_realized_pnl = 0.0

    lp_values: list[float] = []
    hedge_pnls: list[float] = []
    cum_profits: list[float] = []

    last_rebalance_idx = 0
    prev_price = P0

    steps_per_rebalance = max(1, rebalance_minutes // 15)  # 1 step = 15 minutes

    for i, (t, price) in enumerate(zip(index, prices)):
        price = float(price)

        # Update LP price
        lp.update_price(price)

        # Optional: recenter range if price drifts too far from center
        range_center = (lp.lower + lp.upper) / 2.0
        if abs(price - range_center) > range_width * price * 0.5:
            lp.update_range(price, range_width)

        # LP value at this price
        lp_value = lp.get_value_usd()

        # Fees this step
        fee_step = float(fees_per_step.loc[t]) if t in fees_per_step.index else 0.0
        total_fees += fee_step

        # Arbitrage loss proxy
        arb_step = calculate_arbitrage_loss(
            price_prev=prev_price,
            price_curr=price,
            lower=lp.lower,
            upper=lp.upper,
            lp_value_prev=lp_value,
        )
        total_arb_loss += arb_step

        # Hedge
        hedge.update_price(price)

        if (i - last_rebalance_idx) >= steps_per_rebalance:
            # Realize P&L from current hedge
            if abs(hedge.size) > 1e-6:
                realized_pnl = hedge.get_pnl()
                hedge_realized_pnl += realized_pnl

            # Target hedge = - LP delta
            target_delta = lp.get_delta()
            target_hedge_size = -target_delta

            trade_size = abs(target_hedge_size - hedge.size)
            if trade_size > 1e-6:
                hedge.add_trading_cost(
                    trade_size,
                    trading_cost_per_notional,  # cost_bps interpreted as fraction
                )
                total_trading_costs = hedge.cumulative_trading_costs

            hedge.size = target_hedge_size
            hedge.entry_price = price

            last_rebalance_idx = i

        # Track series
        unrealized_hedge_pnl = hedge.get_pnl()
        total_hedge_pnl = hedge_realized_pnl + unrealized_hedge_pnl

        net_lp_component = lp_value - initial_tvl
        cumulative_profit = (
            net_lp_component
            + total_hedge_pnl
            + total_fees
            - total_arb_loss
            - total_trading_costs
        )

        lp_values.append(lp_value)
        hedge_pnls.append(total_hedge_pnl)
        cum_profits.append(cumulative_profit)

        prev_price = price

    final_lp_value = lp_values[-1]
    final_hedge_pnl = hedge_pnls[-1]
    net_profit = (
        final_lp_value
        - initial_tvl
        + final_hedge_pnl
        + total_fees
        - total_arb_loss
        - total_trading_costs
    )
    net_return_pct = (net_profit / initial_tvl) * 100.0

    # APY based on calendar time
    n_days = (index[-1] - index[0]).days or 1
    net_apy = (1.0 + net_profit / initial_tvl) ** (365.0 / n_days) - 1.0
    net_apy *= 100.0

    result = DeltaNeutralResult(
        range_width=range_width,
        rebalance_freq=rebalance_minutes,
        swap_fee=FEE_RATE,
        perp_cost_bps=trading_cost_per_notional,
        initial_tvl=initial_tvl,
        final_lp_value=final_lp_value,
        final_hedge_pnl=final_hedge_pnl,
        total_fees_earned=total_fees,
        total_arbitrage_loss=total_arb_loss,
        total_trading_costs=total_trading_costs,
        total_funding=0.0,
        net_profit=net_profit,
        net_return_pct=net_return_pct,
        net_apy=net_apy,
        lp_values=lp_values,
        hedge_pnls=hedge_pnls,
        cumulative_profits=cum_profits,
    )
    return result


# === MAIN ======================================================================

def main() -> None:
    print("=" * 70)
    print("Aerodrome Pool Delta-Neutral Backtest (CSV inputs)")
    print("=" * 70)

    # 1) ETH price
    print(f"Loading ETH price data from {ETH_CSV} ...")
    price_15m = load_eth_price_15m(ETH_CSV)
    print(f"Price points (15m): {len(price_15m):,}")
    print(f"Price range: ${price_15m.min():.2f} - ${price_15m.max():.2f}")

    # 2) Volume -> daily USD volume + fees
    print(f"\nLoading daily volume from {VOLUME_CSV} ...")
    daily_volume_usd, daily_fee_usd = load_daily_volume_and_fees(price_15m, VOLUME_CSV, FEE_RATE)
    print(f"Total daily USD volume over period: ${daily_volume_usd.sum():,.2f}")
    print(f"Total fee USD over period:          ${daily_fee_usd.sum():,.2f}")

    # 3) Spread fees across 15m steps
    fees_per_step = build_fees_per_step(price_15m, daily_fee_usd)

    # 4) Constant TVL series
    tvl_15m = pd.Series(INITIAL_TVL, index=price_15m.index, name="tvl_usd")
    print(f"\nUsing constant TVL = ${INITIAL_TVL:,.2f}")

    # 5) Run backtest
    print("\nRunning delta-neutral backtest ...")
    result = run_backtest(
        price_15m=price_15m,
        fees_per_step=fees_per_step,
        range_width=RANGE_WIDTH_PCT,
        rebalance_minutes=REBALANCE_MINUTES,
        initial_tvl=INITIAL_TVL,
        trading_cost_per_notional=TRADING_COST_PER_NOTIONAL,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(result)
    print()
    print("Breakdown:")
    print(f"  Initial TVL:        ${result.initial_tvl:,.2f}")
    print(f"  Final LP value:     ${result.final_lp_value:,.2f}")
    print(f"  LP value change:    ${result.final_lp_value - result.initial_tvl:,.2f}")
    print(f"  Hedge P&L:          ${result.final_hedge_pnl:,.2f}")
    print(f"  Fees earned:        ${result.total_fees_earned:,.2f}")
    print(f"  Arbitrage losses:   ${result.total_arbitrage_loss:,.2f}")
    print(f"  Trading costs:      ${result.total_trading_costs:,.2f}")
    print(f"  Funding payments:   ${result.total_funding:,.2f}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  NET PROFIT:         ${result.net_profit:,.2f}")
    print(f"  NET RETURN:         {result.net_return_pct:.2f}%")
    print(f"  NET APY:            {result.net_apy:.2f}%")
    print()

    # 6) Plots in same style as original
    print("Generating plots ...")
    prices_np = price_15m.values
    timestamps_np = price_15m.index.to_numpy()
    out_path = DESKTOP_DIR / "aerodrome_backtest_results.png"
    plot_results(prices_np, timestamps_np, result, output_file=str(out_path))
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()


