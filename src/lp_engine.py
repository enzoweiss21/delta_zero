#!/usr/bin/env python3
"""Delta-neutral LP backtesting engine for dynamic E-CLPs with perpetual hedging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from typing import Optional


# === CONFIGURATION =============================================================

# Default parameters matching typical E-CLP pools
DEFAULT_RANGE_WIDTH: float = 0.10  # ±10% range (e.g., [P*0.9, P*1.1])
DEFAULT_SWAP_FEE: float = 0.0030  # 30 bps (0.30%)
DEFAULT_PERP_TRADING_COST: float = 0.0006  # 6 bps per trade
DEFAULT_INITIAL_TVL: float = 1_000_000.0  # $1M initial TVL

# Rebalancing frequency (in minutes)
# Lower = more frequent hedging (better delta tracking, higher costs)
DEFAULT_REBALANCE_FREQ: int = 15  # Rebalance every 15 minutes

# Volume model parameters
# Realistic volume: typically 0.1-0.5x TVL per day for active pools
# This translates to ~0.004-0.02x TVL per hour, or ~0.00007-0.0003x TVL per minute
DEFAULT_DAILY_VOLUME_MULT: float = 0.2  # Daily volume = 0.2x TVL (conservative)


# === DATA STRUCTURES ===========================================================

@dataclass
class LPPosition:
    """Represents an LP position in a Uniswap v3-style pool."""
    lower: float
    upper: float
    liquidity: float  # L (liquidity units)
    price: float  # Current price (token1 per token0, e.g., USDC per ETH)

    def get_amounts(self) -> tuple[float, float]:
        """
        Get (amount0, amount1) at current price.
        amount0 = ETH, amount1 = USDC
        """
        sqrtP = np.sqrt(self.price)
        sqrtPa = np.sqrt(self.lower)
        sqrtPb = np.sqrt(self.upper)

        if self.price <= self.lower:
            # Fully in token0 (ETH)
            amount0 = self.liquidity * (np.sqrt(self.upper) - np.sqrt(self.lower)) / (sqrtPa * sqrtPb)
            amount1 = 0.0
        elif self.price >= self.upper:
            # Fully in token1 (USDC)
            amount0 = 0.0
            amount1 = self.liquidity * (sqrtPb - sqrtPa)
        else:
            # Mixed
            amount0 = self.liquidity * (sqrtPb - sqrtP) / (sqrtP * sqrtPb)
            amount1 = self.liquidity * (sqrtP - sqrtPa)

        return float(amount0), float(amount1)

    def get_value_usd(self) -> float:
        """Get total value in USD (assuming price is USDC per ETH)."""
        amount0, amount1 = self.get_amounts()
        return amount0 * self.price + amount1

    def get_delta(self) -> float:
        """
        Get ETH exposure (delta) of the position.
        Delta = amount0 (ETH held) - this is what we hedge with perps.
        """
        amount0, _ = self.get_amounts()
        return float(amount0)

    def update_price(self, new_price: float) -> None:
        """Update the current price (pool composition changes)."""
        self.price = new_price

    def update_range(self, new_price: float, width: float) -> None:
        """
        Update the range to center around new_price with given width.
        This simulates a dynamic E-CLP that moves with price.
        """
        k = 1.0 + width
        self.lower = new_price / k
        self.upper = new_price * k
        self.price = new_price


@dataclass
class PerpHedge:
    """Tracks a perpetual futures hedge position."""
    size: float  # Short position size in ETH (negative = short)
    entry_price: float
    current_price: float
    cumulative_funding: float = 0.0
    cumulative_trading_costs: float = 0.0

    def update_price(self, new_price: float) -> None:
        """Update current price for P&L calculation."""
        self.current_price = new_price

    def get_pnl(self) -> float:
        """
        Get unrealized P&L in USD.
        
        Standard futures P&L formula: P&L = (current_price - entry_price) * position_size
        Where position_size is positive for long, negative for short.
        
        Examples:
        - Short 100 ETH (size=-100), entry=1000, current=1100:
          P&L = (1100-1000) * (-100) = -10,000 (loss) ✓
        - Short 100 ETH (size=-100), entry=1000, current=900:
          P&L = (900-1000) * (-100) = 10,000 (profit) ✓
        """
        return (self.current_price - self.entry_price) * self.size

    def get_total_pnl(self) -> float:
        """Get total P&L including funding and costs."""
        return self.get_pnl() + self.cumulative_funding - self.cumulative_trading_costs

    def add_funding(self, funding_rate: float, notional: float) -> None:
        """Add funding payment (typically very small, ~0)."""
        # Funding is paid by shorts to longs (or vice versa)
        # For simplicity, assume funding ≈ 0 (as mentioned in paper)
        self.cumulative_funding += funding_rate * notional * abs(self.size)

    def add_trading_cost(self, trade_size: float, cost_bps: float) -> None:
        """Add trading cost for rebalancing."""
        cost = cost_bps * trade_size * self.current_price
        self.cumulative_trading_costs += cost


@dataclass
class DeltaNeutralResult:
    """Results from a delta-neutral LP backtest."""
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
    net_apy: float  # Annualized

    # Time series (optional, for detailed analysis)
    lp_values: list[float] = field(default_factory=list)
    hedge_pnls: list[float] = field(default_factory=list)
    cumulative_profits: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"DeltaNeutralResult(\n"
            f"  range_width={self.range_width:.3f}, "
            f"rebalance_freq={self.rebalance_freq}min,\n"
            f"  final_lp_value=${self.final_lp_value:,.2f}, "
            f"hedge_pnl=${self.final_hedge_pnl:,.2f},\n"
            f"  fees=${self.total_fees_earned:,.2f}, "
            f"arb_loss=${self.total_arbitrage_loss:,.2f},\n"
            f"  net_profit=${self.net_profit:,.2f} "
            f"({self.net_return_pct:.2f}%), "
            f"APY={self.net_apy:.2f}%\n"
            f")"
        )


# === CORE MATH =================================================================

def estimate_organic_volume(
    price: float,
    lower: float,
    upper: float,
    tvl: float,
    daily_volume_mult: float = DEFAULT_DAILY_VOLUME_MULT,
    time_step_hours: float = 1.0 / 60.0,  # Default: 1 minute = 1/60 hour
    price_volatility: float = 0.0,  # Optional: scale with volatility
) -> float:
    """
    Estimate organic trading volume for this time step.
    
    Uses a simple model: daily volume = daily_volume_mult × TVL
    Then scales down to the time step.
    
    Volume is reduced when pool is out of range (less attractive to traders).
    """
    if price <= lower or price >= upper:
        # Out of range: minimal volume (maybe 10% of normal)
        volume_mult = 0.1
    else:
        # In range: full volume
        volume_mult = 1.0
    
    # Base daily volume
    daily_volume = daily_volume_mult * tvl
    
    # Scale to time step (convert hours to days)
    time_step_days = time_step_hours / 24.0
    base_volume = daily_volume * time_step_days
    
    # Apply range multiplier
    volume = base_volume * volume_mult
    
    # Optional: scale with volatility (more volatility = more trading)
    if price_volatility > 0:
        volume *= (1.0 + price_volatility * 2.0)  # Up to 2x for high vol
    
    return float(volume)


def calculate_arbitrage_loss(
    price_prev: float,
    price_curr: float,
    lower: float,
    upper: float,
    lp_value_prev: float,
) -> float:
    """
    Estimate losses from arbitrage when price moves outside range.
    
    When price moves from inside to outside range, arbitrageurs can:
    1. Swap at better prices than the pool offers
    2. Extract value from LPs
    
    Loss is proportional to how far price moved and the liquidity at the edge.
    """
    was_in_range = lower < price_prev < upper
    is_in_range = lower < price_curr < upper
    
    if was_in_range and not is_in_range:
        # Price moved outside range: arbitrage loss
        # Calculate how far outside the range
        if price_curr < lower:
            distance_pct = (lower - price_curr) / lower
        else:  # price_curr > upper
            distance_pct = (price_curr - upper) / upper
        
        # Loss is typically 0.1-0.3% of TVL per 1% move outside range
        # Capped at reasonable maximum
        loss_pct = min(0.003, distance_pct * 0.002)  # Max 0.3% loss
        loss = lp_value_prev * loss_pct
        return float(loss)
    
    # Also account for continuous arbitrage when price is near range edges
    # (even when still in range, there's some adverse selection)
    if is_in_range:
        # Distance from center
        center = (lower + upper) / 2.0
        range_width = upper - lower
        distance_from_center = abs(price_curr - center) / range_width
        
        # Small continuous loss when near edges (0.01-0.05% of TVL)
        if distance_from_center > 0.7:  # Within 30% of edge
            edge_loss_pct = (distance_from_center - 0.7) * 0.0001  # Very small
            loss = lp_value_prev * edge_loss_pct
            return float(loss)
    
    return 0.0


def _interpolate_real_data(
    timestamps: np.ndarray,
    data_df: pd.DataFrame,
    value_column: str,
    default_value: float = 0.0,
) -> np.ndarray:
    """
    Interpolate real data (from Dune) to match price timestamps.
    
    Args:
        timestamps: Array of timestamps to match
        data_df: DataFrame with 'date' and value_column
        value_column: Name of column with values
        default_value: Value to use if no data available
    
    Returns:
        Array of values matching timestamps (interpolated/forward-filled)
    """
    if data_df.empty or value_column not in data_df.columns:
        return np.full(len(timestamps), default_value)
    
    # Create a series indexed by date
    data_series = data_df.set_index('date')[value_column]
    
    # Convert timestamps to datetime if needed
    if isinstance(timestamps[0], (str, pd.Timestamp)):
        ts_dates = pd.to_datetime(timestamps)
    else:
        # Assume numeric timestamps (Unix time)
        ts_dates = pd.to_datetime(timestamps, unit='s')
    
    # Reindex and forward-fill, then interpolate
    result = data_series.reindex(ts_dates, method='ffill')
    result = result.interpolate(method='linear', limit_direction='both')
    result = result.fillna(default_value)
    
    return result.values


# === MAIN SIMULATION ===========================================================

def simulate_delta_neutral_lp(
    prices: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    range_width: float = DEFAULT_RANGE_WIDTH,
    rebalance_freq: int = DEFAULT_REBALANCE_FREQ,
    swap_fee: float = DEFAULT_SWAP_FEE,
    perp_cost_bps: float = DEFAULT_PERP_TRADING_COST,
    initial_tvl: float = DEFAULT_INITIAL_TVL,
    daily_volume_mult: float = DEFAULT_DAILY_VOLUME_MULT,
    funding_rate: float = 0.0,  # Typically ~0, can be made dynamic
    store_series: bool = False,
    # Real data from Dune (optional - if provided, overrides volume model)
    real_volume_df: Optional[pd.DataFrame] = None,  # Columns: date, volume_usd
    real_tvl_df: Optional[pd.DataFrame] = None,     # Columns: date, tvl_usd
    real_fees_df: Optional[pd.DataFrame] = None,     # Columns: date, fees_usd
) -> DeltaNeutralResult:
    """
    Simulate a delta-neutral LP strategy with dynamic range and perpetual hedging.
    
    Args:
        prices: Array of ETH prices (USDC per ETH)
        timestamps: Optional timestamps for time-based calculations
        range_width: Pool range width (e.g., 0.10 = ±10%)
        rebalance_freq: Rebalance hedge every N minutes
        swap_fee: Pool swap fee (e.g., 0.003 = 30 bps)
        perp_cost_bps: Perp trading cost in basis points
        initial_tvl: Initial total value locked
        volume_per_ld: Volume per unit liquidity density per hour
        funding_rate: Perp funding rate (typically ~0)
        store_series: Whether to store time series data
    
    Returns:
        DeltaNeutralResult with all metrics
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 price points for simulation")
    
    # Initialize LP position
    P0 = float(prices[0])
    k = 1.0 + range_width
    lower0 = P0 / k
    upper0 = P0 * k
    
    # Calculate initial liquidity to match TVL
    # For a position at price P with range [Pa, Pb], value = L * (sqrt(Pb) - sqrt(Pa))
    # when fully in token1. We'll use a simplified approach.
    initial_liquidity = initial_tvl / (np.sqrt(upper0) - np.sqrt(lower0))
    
    lp = LPPosition(
        lower=lower0,
        upper=upper0,
        liquidity=float(initial_liquidity),
        price=P0,
    )
    
    # Initialize hedge (no position initially)
    hedge = PerpHedge(
        size=0.0,
        entry_price=P0,
        current_price=P0,
    )
    
    # Prepare real data if available (interpolate to match price timestamps)
    use_real_volume = real_volume_df is not None and not real_volume_df.empty
    use_real_fees = real_fees_df is not None and not real_fees_df.empty
    use_real_tvl = real_tvl_df is not None and not real_tvl_df.empty
    
    if use_real_volume:
        # Interpolate real volume to match price timestamps
        if timestamps is not None:
            real_volume_series = _interpolate_real_data(
                timestamps, real_volume_df, 'volume_usd', default_value=0.0
            )
        else:
            # No timestamps - can't match, fall back to model
            use_real_volume = False
            print("Warning: Real volume data provided but no timestamps. Using volume model.")
    
    if use_real_fees:
        if timestamps is not None:
            real_fees_series = _interpolate_real_data(
                timestamps, real_fees_df, 'fees_usd', default_value=0.0
            )
        else:
            use_real_fees = False
            print("Warning: Real fees data provided but no timestamps. Using fee model.")
    
    if use_real_tvl:
        if timestamps is not None:
            real_tvl_series = _interpolate_real_data(
                timestamps, real_tvl_df, 'tvl_usd', default_value=initial_tvl
            )
        else:
            use_real_tvl = False
    
    # Trackers
    total_fees = 0.0
    total_arb_loss = 0.0
    cumulative_hedge_pnl = 0.0  # Track realized + unrealized hedge P&L
    last_rebalance_idx = 0
    
    # Time series (if requested)
    lp_values_series = [initial_tvl] if store_series else []
    hedge_pnls_series = [0.0] if store_series else []
    cumulative_profits_series = [0.0] if store_series else []
    
    # Main simulation loop
    for i in range(1, len(prices)):
        P_prev = float(prices[i - 1])
        P_curr = float(prices[i])
        
        # Update LP price (composition changes)
        lp.update_price(P_curr)
        
        # Update pool range if needed (dynamic E-CLP moves with price)
        # Check if price is near range edges and adjust
        range_center = (lp.lower + lp.upper) / 2.0
        if abs(P_curr - range_center) > range_width * P_curr * 0.5:
            # Price moved significantly, update range
            lp.update_range(P_curr, range_width)
        
        # Calculate current LP value
        lp_value = lp.get_value_usd()
        
        # Get volume and fees - use real data if available, otherwise model
        if use_real_volume:
            # Use real volume from Dune
            volume = float(real_volume_series[i])
        else:
            # Estimate volume using model
            if i > 10:
                recent_prices = prices[max(0, i-10):i+1]
                price_vol = float(np.std(recent_prices) / np.mean(recent_prices))
            else:
                price_vol = 0.0
            
            volume = estimate_organic_volume(
                P_curr, lp.lower, lp.upper, lp_value,
                daily_volume_mult=daily_volume_mult,
                time_step_hours=1.0 / 60.0,  # 1 minute steps
                price_volatility=price_vol,
            )
        
        # Calculate fees - use real fees if available, otherwise calculate from volume
        if use_real_fees:
            # Use real fees from Dune (already in USD)
            fees_this_step = float(real_fees_series[i] - (real_fees_series[i-1] if i > 0 else 0))
            if fees_this_step < 0:
                fees_this_step = 0  # Don't allow negative fees
        else:
            # Calculate fees from volume
            fees_this_step = swap_fee * volume
        
        total_fees += fees_this_step
        
        # Update TVL if real TVL data available
        if use_real_tvl:
            # Use real TVL for volume calculations
            current_tvl = float(real_tvl_series[i])
            lp_value = current_tvl  # Update LP value to match real TVL
        
        # Calculate arbitrage loss
        arb_loss = calculate_arbitrage_loss(
            P_prev, P_curr, lp.lower, lp.upper, lp_value
        )
        total_arb_loss += arb_loss
        
        # Update hedge price
        hedge.update_price(P_curr)
        
        # Rebalance hedge if needed
        should_rebalance = (i - last_rebalance_idx) >= rebalance_freq
        if should_rebalance or i == len(prices) - 1:  # Always rebalance at end
            # Realize P&L from current hedge position before rebalancing
            if abs(hedge.size) > 1e-6:
                realized_pnl = hedge.get_pnl()  # P&L since last entry
                cumulative_hedge_pnl += realized_pnl
            
            # Calculate required hedge size (short ETH to offset LP delta)
            target_delta = lp.get_delta()
            current_hedge_size = hedge.size
            
            # We want: LP_delta + hedge_size = 0 (delta neutral)
            # So: hedge_size = -LP_delta
            target_hedge_size = -target_delta
            
            # Calculate trade size
            trade_size = abs(target_hedge_size - current_hedge_size)
            
            if trade_size > 1e-6:  # Only trade if meaningful
                # Pay trading cost
                hedge.add_trading_cost(trade_size, perp_cost_bps)
            
            # Update hedge position (reset entry price for new position)
            hedge.size = target_hedge_size
            hedge.entry_price = P_curr
            
            # Add funding (typically ~0)
            if abs(hedge.size) > 1e-6:
                notional = abs(hedge.size) * P_curr
                hedge.add_funding(funding_rate, notional)
            
            last_rebalance_idx = i
        
        # Store time series if requested
        if store_series:
            lp_values_series.append(lp_value)
            hedge_pnls_series.append(hedge.get_pnl())
            cumulative_profits_series.append(
                lp_value - initial_tvl + hedge.get_total_pnl() + total_fees - total_arb_loss
            )
    
    # Final calculations
    final_lp_value = lp.get_value_usd()
    
    # Add final unrealized P&L
    final_unrealized_pnl = hedge.get_pnl() if abs(hedge.size) > 1e-6 else 0.0
    total_hedge_pnl = cumulative_hedge_pnl + final_unrealized_pnl
    
    net_profit = (
        final_lp_value - initial_tvl +  # LP value change
        total_hedge_pnl +  # Hedge P&L (realized + unrealized)
        total_fees -  # Fees earned
        total_arb_loss  # Arbitrage losses
    )
    
    net_return_pct = (net_profit / initial_tvl) * 100.0
    
    # Calculate APY (annualized)
    # Assume prices are 1-minute intervals
    num_minutes = len(prices) - 1
    num_years = num_minutes / (365.25 * 24 * 60)
    if num_years > 0:
        net_apy = ((1.0 + net_profit / initial_tvl) ** (1.0 / num_years) - 1.0) * 100.0
    else:
        net_apy = 0.0
    
    return DeltaNeutralResult(
        range_width=range_width,
        rebalance_freq=rebalance_freq,
        swap_fee=swap_fee,
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
        lp_values=lp_values_series if store_series else [],
        hedge_pnls=hedge_pnls_series if store_series else [],
        cumulative_profits=cumulative_profits_series if store_series else [],
    )


# === OPTIMIZATION ==============================================================

def optimize_delta_neutral_lp(
    prices: np.ndarray,
    range_widths: Iterable[float] = (0.05, 0.10, 0.15, 0.20),
    rebalance_freqs: Iterable[int] = (5, 15, 30, 60),
    swap_fee: float = DEFAULT_SWAP_FEE,
    perp_cost_bps: float = DEFAULT_PERP_TRADING_COST,
    initial_tvl: float = DEFAULT_INITIAL_TVL,
    daily_volume_mult: float = DEFAULT_DAILY_VOLUME_MULT,
) -> tuple[DeltaNeutralResult, list[DeltaNeutralResult]]:
    """
    Grid search over range widths and rebalance frequencies to find optimal strategy.
    
    Returns:
        (best_result, all_results)
    """
    results: list[DeltaNeutralResult] = []
    
    for width in range_widths:
        for freq in rebalance_freqs:
            result = simulate_delta_neutral_lp(
                prices=prices,
                range_width=width,
                rebalance_freq=freq,
                swap_fee=swap_fee,
                perp_cost_bps=perp_cost_bps,
                initial_tvl=initial_tvl,
                daily_volume_mult=daily_volume_mult,
            )
            results.append(result)
    
    # Best by highest APY
    best = max(results, key=lambda r: r.net_apy)
    return best, results

