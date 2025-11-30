#!/usr/bin/env python3
"""Uniswap v3 LP backtesting simulator with automatic optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# === CONFIG ===================================================================

# Adjust this only if your CSV lives somewhere else.
# Right now it assumes: delta_zero/data/eth_1min.csv
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "eth_1min.csv"

# If None, we'll try common column names like "close", "price", etc.
DEFAULT_PRICE_COLUMN: str | None = None

# How often to sample price points (1 = every row, 5 = every 5th row, etc.)
# Using 5 or 10 speeds up long multi-year backtests a LOT.
DOWNSAMPLE_STEP: int = 5

# Default grid of range widths (multiplicative symmetric range).
# Example: 0.5 means [P0/1.5, P0*1.5] ≈ [P0*0.667, P0*1.5].
DEFAULT_WIDTH_GRID: Sequence[float] = (0.3, 0.5, 0.7, 1.0)

# Approximate fee model:
#   fee_per_period ≈ fee_tier * (volume_mult * TVL * |return|)
FEE_TIER: float = 0.0005      # 5 bps, e.g. Uniswap v3 0.05% pool
VOLUME_MULT: float = 5.0      # "volume ≈ 5x TVL * |return|" proxy


# === DATA STRUCTURES ==========================================================

@dataclass
class LPSimulationResult:
    lower: float
    upper: float
    width_fraction: float
    initial_price: float
    final_value_multiple: float    # LP value / initial value (ex fees)
    fees_multiple: float           # fees / initial value
    total_multiple: float          # (value + fees) / initial value

    def __repr__(self) -> str:
        return (
            "LPSimulationResult("
            f"lower={self.lower:.2f}, "
            f"upper={self.upper:.2f}, "
            f"width_fraction={self.width_fraction:.3f}, "
            f"initial_price={self.initial_price:.2f}, "
            f"final_value_multiple={self.final_value_multiple:.4f}, "
            f"fees_multiple={self.fees_multiple:.4f}, "
            f"total_multiple={self.total_multiple:.4f})"
        )


# === UTILITIES ================================================================

def _auto_pick_price_column(df: pd.DataFrame, explicit: str | None) -> str:
    """
    Figure out which column contains the price if user didn't specify.
    Priority:
    1) explicit
    2) common names
    3) second column as fallback
    """
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"Price column '{explicit}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        return explicit

    candidates = ("close", "price", "last", "eth_price", "ETHUSDT")
    for c in candidates:
        if c in df.columns:
            return c

    if len(df.columns) < 2:
        raise ValueError(
            "Could not infer price column. CSV has too few columns. "
            f"Columns: {list(df.columns)}"
        )
    # Fallback: take the second column (first often is a timestamp)
    return df.columns[1]


def get_price_series(
    csv_path: Path = CSV_PATH,
    price_column: str | None = DEFAULT_PRICE_COLUMN,
    downsample_step: int = DOWNSAMPLE_STEP,
) -> np.ndarray:
    """
    Load price series from CSV and return as numpy array of floats.
    Automatically picks a reasonable price column if not specified.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Price file not found at {csv_path}.\n"
            "→ Fix CSV_PATH in lp_simulator.py or create the file there."
        )

    df = pd.read_csv(csv_path)
    col = _auto_pick_price_column(df, price_column)

    prices = df[col].astype(float).to_numpy()

    if downsample_step > 1:
        prices = prices[::downsample_step]

    if prices.size < 3:
        raise ValueError("Price series is too short after loading/downsampling.")

    return prices


# === UNISWAP V3 MATH (STATIC RANGE) ===========================================

def _lp_value_at_price(P: float, Pa: float, Pb: float, L: float = 1.0) -> float:
    """
    Uniswap v3 position value at price P (token1 units),
    with lower = Pa, upper = Pb, and liquidity L.

    Assumes price = token1 per token0.
    """
    sqrtP = np.sqrt(P)
    sqrtPa = np.sqrt(Pa)
    sqrtPb = np.sqrt(Pb)

    if P <= Pa:
        # Fully token0
        amount0 = L * (sqrtPb - sqrtPa) / (sqrtPa * sqrtPb)
        amount1 = 0.0
    elif P >= Pb:
        # Fully token1
        amount0 = 0.0
        amount1 = L * (sqrtPb - sqrtPa)
    else:
        # Mixed
        amount0 = L * (sqrtPb - sqrtP) / (sqrtP * sqrtPb)
        amount1 = L * (sqrtP - sqrtPa)

    return amount0 * P + amount1


def _simulate_lp_for_width(
    prices: np.ndarray,
    width_fraction: float,
    fee_tier: float = FEE_TIER,
    volume_mult: float = VOLUME_MULT,
) -> LPSimulationResult:
    """
    Simulate a static Uniswap v3 LP position with a symmetric range
    around the initial price using a simple fee model.

    Returns metrics normalized to initial LP value (so 1.10 = +10%).
    """
    if width_fraction <= 0:
        raise ValueError("width_fraction must be > 0")

    P0 = float(prices[0])
    # Use multiplicative symmetric range: [P0/k, P0*k] where k = 1 + width_fraction
    # This ensures lower is always positive and works for any width_fraction > 0
    k = 1.0 + width_fraction
    lower = P0 / k
    upper = P0 * k

    if lower <= 0:
        raise ValueError(
            f"Computed lower bound {lower} is not positive. "
            f"Check width_fraction={width_fraction} and price data."
        )

    # Precompute LP values for each price point with L = 1
    values = np.empty_like(prices, dtype=float)
    for i, P in enumerate(prices):
        values[i] = _lp_value_at_price(float(P), lower, upper, L=1.0)

    V0 = float(values[0])
    if V0 <= 0:
        raise ValueError("Initial LP value is non-positive, something is off.")

    # Normalize values by initial LP value
    values /= V0

    # Approximate fee generation over the path
    fees = 0.0
    for i in range(1, len(prices)):
        P_prev = float(prices[i - 1])
        P_curr = float(prices[i])
        V_prev = float(values[i - 1])

        # Only collect fees when inside range at either end
        if lower < P_prev < upper or lower < P_curr < upper:
            ret = abs(P_curr / P_prev - 1.0)
            # Very rough "volume" proxy
            volume = volume_mult * V_prev * ret
            fees += fee_tier * volume

    final_value = float(values[-1])        # already normalized
    total = final_value + fees             # fees already normalized

    return LPSimulationResult(
        lower=lower,
        upper=upper,
        width_fraction=width_fraction,
        initial_price=P0,
        final_value_multiple=final_value,
        fees_multiple=fees,
        total_multiple=total,
    )


def optimize_static_lp(
    prices: np.ndarray,
    width_grid: Iterable[float] = DEFAULT_WIDTH_GRID,
    fee_tier: float = FEE_TIER,
    volume_mult: float = VOLUME_MULT,
) -> tuple[LPSimulationResult, list[LPSimulationResult]]:
    """
    Run a simple grid search over symmetric LP ranges and return:
      (best_result, all_results)

    `best_result` is chosen by highest total_multiple (value + fees).
    """
    results: list[LPSimulationResult] = []

    for w in width_grid:
        res = _simulate_lp_for_width(
            prices=prices,
            width_fraction=float(w),
            fee_tier=fee_tier,
            volume_mult=volume_mult,
        )
        results.append(res)

    best = max(results, key=lambda r: r.total_multiple)
    return best, results


# === PUBLIC API ===============================================================

def run_optimized_lp_backtest(
    csv_path: Path = CSV_PATH,
    price_column: str | None = DEFAULT_PRICE_COLUMN,
    width_grid: Iterable[float] = DEFAULT_WIDTH_GRID,
    downsample_step: int = DOWNSAMPLE_STEP,
    fee_tier: float = FEE_TIER,
    volume_mult: float = VOLUME_MULT,
) -> LPSimulationResult:
    """
    Fully automated:
      1) Load price series from CSV
      2) Try a grid of LP width ranges
      3) Pick the best one by final (value + fees)

    Returns the best LPSimulationResult.
    """
    prices = get_price_series(
        csv_path=csv_path,
        price_column=price_column,
        downsample_step=downsample_step,
    )

    best, all_results = optimize_static_lp(
        prices=prices,
        width_grid=width_grid,
        fee_tier=fee_tier,
        volume_mult=volume_mult,
    )

    # Optional: print a small leaderboard
    print("=== Static LP Grid Search Results (normalized to initial LP value) ===")
    for r in sorted(all_results, key=lambda x: x.total_multiple, reverse=True):
        print(
            f"width=±{r.width_fraction:.2f} "
            f"→ final={r.final_value_multiple:.4f}, "
            f"fees={r.fees_multiple:.4f}, "
            f"total={r.total_multiple:.4f}"
        )
    print("=== Best configuration ===")
    print(best)

    return best


# === SCRIPT ENTRYPOINT ========================================================

if __name__ == "__main__":
    # One-shot, zero-manual run:
    # Uses CSV_PATH, auto price column, default grid, etc.
    result = run_optimized_lp_backtest()
    print("\nFinal best result object:")
    print(result)
