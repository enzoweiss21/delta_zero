#!/usr/bin/env python3
"""Phase 2: Delta-neutral LP backtest using Snowflake CSV inputs.

Data inputs (already in this repo):
  1) Minute ETH prices:  data/eth_1min.csv
       - Columns: timestamp, price
  2) Daily pool volume: data/snowflake/Volume_05:01:2024-10:30:2025.csv
       - Columns: day, token0_volume_raw, token1_volume_raw, total_raw_volume
  3) Liquidity events:  data/snowflake/Liquidity_05:01:2024-10:30:2025.csv
       - Columns: timestamp, event_name (Mint/Burn/Swap), liquidity_delta

This script:
  - Resamples ETH price to 15-minute bars
  - Builds TVL from liquidity events and ETH price
  - Computes daily fee revenue from volume * pool_fee
  - Simulates a range LP + delta-neutral hedge at 15-minute granularity
  - Produces the same style dashboard as the original backtest
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure we can import core engine modules when running as:
#   python src/phase2/phase2_backtest.py
import sys
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lp_engine import (
    LPPosition,
    PerpHedge,
    calculate_arbitrage_loss,
    DEFAULT_REBALANCE_FREQ,
    DEFAULT_PERP_TRADING_COST,
)
from delta_neutral_backtest import plot_results


# === PATHS & POOL CONFIG =======================================================

# Project root is two levels up from this file: .../delta_zero/
ROOT = Path(__file__).resolve().parents[2]

ETH_CSV = ROOT / "data" / "eth_1min.csv"
VOLUME_CSV = ROOT / "data" / "snowflake" / "Volume_05:01:2024-10:30:2025.csv"
LIQUIDITY_CSV = ROOT / "data" / "snowflake" / "Liquidity_05:01:2024-10:30:2025.csv"

POOL_FEE_TIER = 0.000356  # 0.0356%
POOL_START_DATE = pd.Timestamp("2024-05-01")
BACKTEST_END_DATE = pd.Timestamp("2025-01-31")

REBALANCE_INTERVAL_MIN = 15  # 15-minute rebalance grid
RANGE_WIDTH_DEFAULT = 0.10   # ±10%
INITIAL_TVL_DEFAULT = 1_000_000.0  # used only for APY normalization


# === RESULT STRUCTURE ==========================================================

@dataclass
class Phase2BacktestResult:
    """Container for phase 2 backtest results (compatible with plot_results)."""

    # Configuration
    range_width: float
    rebalance_freq: int
    swap_fee: float
    perp_cost_bps: float
    initial_tvl: float

    # Final metrics
    final_lp_value: float
    final_hedge_pnl: float
    total_fees_earned: float
    total_arbitrage_loss: float
    total_trading_costs: float
    total_funding: float
    net_profit: float
    net_return_pct: float
    net_apy: float

    # Time series (for plotting)
    lp_values: list[float] = field(default_factory=list)
    hedge_pnls: list[float] = field(default_factory=list)
    cumulative_profits: list[float] = field(default_factory=list)

    # Additional detail series (pandas, aligned on 15-min index)
    index: pd.DatetimeIndex | None = None
    price_series: pd.Series | None = None
    tvl_series: pd.Series | None = None
    fee_series: pd.Series | None = None

    def to_series_dict(self) -> Dict[str, pd.Series]:
        """Return a dictionary of the main timeseries."""
        if self.index is None:
            return {}
        data = {
            "price": self.price_series,
            "tvl": self.tvl_series,
            "fees_per_step": self.fee_series,
        }
        # Attach numpy-based lists as series on same index
        if self.lp_values:
            data["lp_value_model"] = pd.Series(self.lp_values, index=self.index[: len(self.lp_values)])
        if self.hedge_pnls:
            data["hedge_pnl"] = pd.Series(self.hedge_pnls, index=self.index[: len(self.hedge_pnls)])
        if self.cumulative_profits:
            data["cumulative_profit"] = pd.Series(
                self.cumulative_profits, index=self.index[: len(self.cumulative_profits)]
            )
        return {k: v for k, v in data.items() if v is not None}

    def __repr__(self) -> str:
        return (
            "Phase2BacktestResult(\n"
            f"  range_width={self.range_width:.3f}, rebalance_freq={self.rebalance_freq}min,\n"
            f"  final_lp_value=${self.final_lp_value:,.2f}, hedge_pnl=${self.final_hedge_pnl:,.2f},\n"
            f"  fees=${self.total_fees_earned:,.2f}, arb_loss=${self.total_arbitrage_loss:,.2f},\n"
            f"  net_profit=${self.net_profit:,.2f} ({self.net_return_pct:.2f}%), "
            f"APY={self.net_apy:.2f}%\n"
            ")"
        )


# === DATA LOADING & PREP ======================================================

def load_eth_price_15m(
    csv_path: Path = ETH_CSV,
    start: pd.Timestamp = POOL_START_DATE,
    end: pd.Timestamp = BACKTEST_END_DATE,
    freq: str = "15min",
) -> pd.Series:
    """Load 1-min ETH price CSV and resample to 15-minute close, starting May 1 2024."""
    if not csv_path.exists():
        raise FileNotFoundError(f"ETH price CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Expected 'timestamp' column in {csv_path}, found {list(df.columns)}")
    # Try common price column names
    price_col_candidates = ["price", "close", "ETHUSDT", "eth_price"]
    price_col = None
    for c in price_col_candidates:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        # Fallback: second column
        price_col = df.columns[1]

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df.loc[start : end + pd.Timedelta(days=1)]

    price_15m = df[price_col].resample(freq).last().ffill()
    price_15m.name = "price"
    return price_15m


def load_liquidity_series(
    csv_path: Path = LIQUIDITY_CSV,
    price_15m: Optional[pd.Series] = None,
) -> tuple[pd.Series, pd.Series]:
    """Build liquidity units and TVL over time from Mint/Burn events.

    Returns:
        liquidity_units_15m: pd.Series aligned to price_15m.index
        tvl_15m:             pd.Series aligned to price_15m.index
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Liquidity CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Case-insensitive column lookup to match Snowflake export
    lower_map = {c.lower(): c for c in df.columns}
    # Snowflake uses DAY as the timestamp column here
    ts_col = (
        lower_map.get("timestamp")
        or lower_map.get("time")
        or lower_map.get("ts")
        or lower_map.get("day")
    )
    event_col = lower_map.get("event_name") or lower_map.get("event")
    liq_col = lower_map.get("liquidity_delta") or lower_map.get("liquidity")

    if ts_col is None or event_col is None or liq_col is None:
        raise ValueError(
            "Liquidity CSV must contain timestamp/event_name/liquidity_delta "
            f"(any case). Got columns: {list(df.columns)}"
        )

    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col)
    df = df[df[ts_col] >= POOL_START_DATE]

    # Keep only Mint/Burn; Swaps don't change TVL units (case-insensitive)
    df[event_col] = df[event_col].astype(str)
    df = df[df[event_col].str.lower().isin(["mint", "burn"])]

    # Apply sign convention: Mint increases, Burn decreases
    signed_delta = np.where(
        df[event_col] == "Burn",
        -df[liq_col].astype(float),
        df[liq_col].astype(float),
    )
    df["liquidity_signed"] = signed_delta

    # Set timestamp index and aggregate any duplicate timestamps (Snowflake can emit
    # multiple rows per DAY/event combination).
    df = df.set_index(ts_col)
    # Sum signed deltas per timestamp, then take cumulative sum over time
    liquidity_units = (
        df["liquidity_signed"]
        .sort_index()
        .groupby(level=0)
        .sum()
        .cumsum()
    )

    if price_15m is None:
        # If no price grid given, just return liquidity units at native timestamps
        return liquidity_units, liquidity_units * np.nan

    # Align to 15-min price index
    liquidity_15m = liquidity_units.reindex(price_15m.index, method="ffill").fillna(0.0)
    tvl_15m = liquidity_15m * price_15m  # TVL ≈ liquidity_units * ETH price

    liquidity_15m.name = "liquidity_units"
    tvl_15m.name = "tvl_usd"
    return liquidity_15m, tvl_15m


def load_daily_volume(csv_path: Path = VOLUME_CSV) -> pd.Series:
    """Load daily pool volume from Snowflake CSV and compute daily fee revenue."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Volume CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Case-insensitive mapping for Snowflake headers: DAY, TOTAL_RAW_VOLUME, etc.
    lower_map = {c.lower(): c for c in df.columns}
    day_col = lower_map.get("day")
    total_vol_col = lower_map.get("total_raw_volume")
    if day_col is None or total_vol_col is None:
        raise ValueError(
            "Volume CSV must contain day/total_raw_volume (any case). "
            f"Got columns: {list(df.columns)}"
        )

    df[day_col] = pd.to_datetime(df[day_col]).dt.normalize()
    df = df.sort_values(day_col)

    raw_vol = df[total_vol_col].astype(float)

    # Snowflake export uses raw token units (e.g. 18 decimals). Auto-normalize if needed.
    # Heuristic: if median is huge, divide by 1e18 to get token units.
    median_raw = float(raw_vol.median())
    if median_raw > 1e15:
        volume_token = raw_vol / 1e18
    else:
        volume_token = raw_vol

    df["daily_fee_usd"] = volume_token * POOL_FEE_TIER
    daily_fees = df.set_index(day_col)["daily_fee_usd"]
    daily_fees.name = "daily_fee_usd"
    return daily_fees


def compute_fee_per_step(
    index_15m: pd.DatetimeIndex,
    daily_fees: pd.Series,
) -> pd.Series:
    """Map daily fees into a constant-per-15min fee accrual series."""
    if daily_fees.empty:
        return pd.Series(0.0, index=index_15m, name="fees_per_step")

    # Map each 15-min timestamp to its day, then divide by 96 periods/day
    day_index = pd.to_datetime(index_15m.normalize())
    per_day_fee = daily_fees.reindex(day_index.unique()).fillna(0.0)
    fee_for_ts = day_index.map(per_day_fee) / 96.0

    fee_series = pd.Series(fee_for_ts.values, index=index_15m, name="fees_per_step")
    return fee_series


# === CORE SIMULATION ===========================================================

def _lp_unit_value(lower: float, upper: float, price: float) -> float:
    """Value of LP position with L=1 at given price (token1 USD units)."""
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


def simulate_phase2_lp(
    price_15m: pd.Series,
    tvl_15m: pd.Series,
    fee_per_step: pd.Series,
    range_width: float = RANGE_WIDTH_DEFAULT,
    rebalance_freq: int = REBALANCE_INTERVAL_MIN,
    perp_cost_bps: float = DEFAULT_PERP_TRADING_COST,
    funding_rate: float = 0.0,
    store_series: bool = True,
) -> Phase2BacktestResult:
    """Simulate a delta-neutral LP using real TVL, fees, and 15-min prices."""
    if len(price_15m) < 2:
        raise ValueError("Need at least 2 price points for simulation")

    # Ensure all series share the same index
    index = price_15m.index
    tvl_15m = tvl_15m.reindex(index).ffill()
    fee_per_step = fee_per_step.reindex(index).fillna(0.0)

    P0 = float(price_15m.iloc[0])
    initial_tvl = float(tvl_15m.iloc[0])
    if initial_tvl <= 0:
        # Fallback to a notional TVL if liquidity CSV is empty or zeroed
        initial_tvl = INITIAL_TVL_DEFAULT
        tvl_15m = pd.Series(initial_tvl, index=index, name="tvl_usd")

    k = 1.0 + range_width
    lower0 = P0 / k
    upper0 = P0 * k

    unit_val0 = _lp_unit_value(lower0, upper0, P0)
    if unit_val0 <= 0:
        raise ValueError("Initial LP unit value is non-positive, check inputs.")
    initial_liquidity = initial_tvl / unit_val0

    lp = LPPosition(lower=lower0, upper=upper0, liquidity=float(initial_liquidity), price=P0)
    hedge = PerpHedge(size=0.0, entry_price=P0, current_price=P0)

    total_fees = 0.0
    total_arb_loss = 0.0
    cumulative_hedge_pnl = 0.0
    last_rebalance_idx = 0

    lp_values_series: list[float] = []
    hedge_pnls_series: list[float] = []
    cumulative_profits_series: list[float] = []

    for i in range(1, len(index)):
        ts_prev = index[i - 1]
        ts_curr = index[i]
        P_prev = float(price_15m.iloc[i - 1])
        P_curr = float(price_15m.iloc[i])

        # Move LP price
        lp.update_price(P_curr)

        # Adjust range if price drifts away from center (dynamic E-CLP style)
        range_center = (lp.lower + lp.upper) / 2.0
        if abs(P_curr - range_center) > range_width * P_curr * 0.5:
            lp.update_range(P_curr, range_width)

        # Adjust liquidity to match external TVL at this timestamp
        target_tvl = float(tvl_15m.loc[ts_curr])
        unit_val = _lp_unit_value(lp.lower, lp.upper, P_curr)
        if unit_val <= 0:
            # Fallback: keep previous liquidity if unit value breaks
            lp_value = lp.get_value_usd()
        else:
            lp.liquidity = target_tvl / unit_val
            lp_value = lp.get_value_usd()

        # Fees this step (we assume we own 100% of liquidity)
        fees_this_step = float(fee_per_step.loc[ts_curr])
        total_fees += fees_this_step

        # Arbitrage loss proxy
        lp_value_prev = float(tvl_15m.loc[ts_prev])
        arb_loss = calculate_arbitrage_loss(P_prev, P_curr, lp.lower, lp.upper, lp_value_prev)
        total_arb_loss += arb_loss

        # Hedge dynamics
        hedge.update_price(P_curr)

        minutes_since_last = (i - last_rebalance_idx) * REBALANCE_INTERVAL_MIN
        should_rebalance = minutes_since_last >= rebalance_freq or i == len(index) - 1

        if should_rebalance:
            # Realize hedge P&L from previous position
            if abs(hedge.size) > 1e-6:
                realized_pnl = hedge.get_pnl()
                cumulative_hedge_pnl += realized_pnl

            # Target hedge = - delta of LP
            target_delta = lp.get_delta()
            current_hedge_size = hedge.size
            target_hedge_size = -target_delta

            trade_size = abs(target_hedge_size - current_hedge_size)
            if trade_size > 1e-6:
                hedge.add_trading_cost(trade_size, perp_cost_bps)

            hedge.size = target_hedge_size
            hedge.entry_price = P_curr

            if abs(hedge.size) > 1e-6:
                notional = abs(hedge.size) * P_curr
                hedge.add_funding(funding_rate, notional)

            last_rebalance_idx = i

        if store_series:
            lp_values_series.append(lp_value)
            hedge_pnls_series.append(hedge.get_pnl())
            cumulative_profits_series.append(
                lp_value
                - initial_tvl
                + hedge.get_total_pnl()
                + total_fees
                - total_arb_loss
            )

    final_lp_value = float(tvl_15m.iloc[-1])
    final_unrealized_pnl = hedge.get_pnl() if abs(hedge.size) > 1e-6 else 0.0
    total_hedge_pnl = cumulative_hedge_pnl + final_unrealized_pnl

    net_profit = (
        final_lp_value
        - initial_tvl
        + total_hedge_pnl
        + total_fees
        - total_arb_loss
    )
    if initial_tvl > 0:
        net_return_pct = (net_profit / initial_tvl) * 100.0
    else:
        net_return_pct = 0.0

    # APY based on 15-min grid
    total_minutes = len(index) * REBALANCE_INTERVAL_MIN
    num_years = total_minutes / (365.25 * 24 * 60)
    if num_years > 0 and initial_tvl > 0:
        net_apy = ((1.0 + net_profit / initial_tvl) ** (1.0 / num_years) - 1.0) * 100.0
    else:
        net_apy = 0.0

    result = Phase2BacktestResult(
        range_width=range_width,
        rebalance_freq=rebalance_freq,
        swap_fee=POOL_FEE_TIER,
        perp_cost_bps=perp_cost_bps,
        initial_tvl=initial_tvl,
        final_lp_value=final_lp_value,
        final_hedge_pnl=total_hedge_pnl,
        total_fees_earned=total_fees,
        total_arbitrage_loss=total_arb_loss,
        total_trading_costs=hedge.cumulative_trading_costs,
        total_funding=hedge.cumulative_funding,
        net_profit=net_profit,
        net_return_pct=net_return_pct,
        net_apy=net_apy,
        lp_values=lp_values_series,
        hedge_pnls=hedge_pnls_series,
        cumulative_profits=cumulative_profits_series,
        index=index,
        price_series=price_15m,
        tvl_series=tvl_15m,
        fee_series=fee_per_step,
    )
    return result


# === ORCHESTRATION & CLI =======================================================

def run_phase2_backtest(
    range_width: float = RANGE_WIDTH_DEFAULT,
    rebalance_freq: int = DEFAULT_REBALANCE_FREQ,
    plot: bool = True,
    plot_file: Optional[str] = "phase2_backtest_results.png",
) -> Phase2BacktestResult:
    """End-to-end runner: load data, simulate, and plot."""
    print("=" * 70)
    print("Phase 2 Delta-Neutral LP Backtest (Snowflake data)")
    print("=" * 70)

    # 1) Prices
    print(f"Loading ETH price data from {ETH_CSV.name}...")
    price_15m = load_eth_price_15m()
    print(f"Price points (15m): {len(price_15m):,}")
    print(f"Price range: ${price_15m.min():.2f} - ${price_15m.max():.2f}")

    # 2) Liquidity → TVL
    print(f"\nLoading liquidity events from {LIQUIDITY_CSV.name}...")
    liquidity_15m, tvl_15m = load_liquidity_series(LIQUIDITY_CSV, price_15m)
    print(f"TVL range: ${tvl_15m.min():.2f} - ${tvl_15m.max():.2f}")
    if tvl_15m.max() <= 0:
        print("Warning: TVL series is zero; falling back to notional TVL "
              f"${INITIAL_TVL_DEFAULT:,.0f} for simulation.")

    # 3) Volume → daily fees → per-step fees
    print(f"\nLoading daily volume from {VOLUME_CSV.name}...")
    daily_fees = load_daily_volume(VOLUME_CSV)
    fee_per_step = compute_fee_per_step(price_15m.index, daily_fees)
    print(f"Total fees over period: ${daily_fees.sum():,.2f}")

    # 4) Simulation
    print("\nRunning phase 2 simulation...")
    result = simulate_phase2_lp(
        price_15m=price_15m,
        tvl_15m=tvl_15m,
        fee_per_step=fee_per_step,
        range_width=range_width,
        rebalance_freq=rebalance_freq,
        perp_cost_bps=DEFAULT_PERP_TRADING_COST,
        funding_rate=0.0,
        store_series=True,
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

    if plot:
        print("Generating plots (phase 2 style matches original)...")
        # Reuse original plotting layout/colors
        prices_np = price_15m.values
        timestamps_np = price_15m.index.to_numpy()
        plot_results(prices_np, timestamps_np, result, output_file=plot_file)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 2 Delta-Neutral LP Backtest (Snowflake data)",
    )
    parser.add_argument(
        "--range-width",
        type=float,
        default=RANGE_WIDTH_DEFAULT,
        help=f"Range width (default: {RANGE_WIDTH_DEFAULT})",
    )
    parser.add_argument(
        "--rebalance-freq",
        type=int,
        default=DEFAULT_REBALANCE_FREQ,
        help=f"Hedge rebalance frequency in minutes (default: {DEFAULT_REBALANCE_FREQ})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run without generating plots",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default="phase2_backtest_results.png",
        help="Output PNG file for plots",
    )

    args = parser.parse_args()
    run_phase2_backtest(
        range_width=args.range_width,
        rebalance_freq=args.rebalance_freq,
        plot=not args.no_plot,
        plot_file=args.plot_file,
    )


