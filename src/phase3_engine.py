#!/usr/bin/env python3
"""Phase 3: Automated LP Strategy with Edge-Rebalance and Real Dune Data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# === CONFIGURATION =============================================================

DEFAULT_RANGE_WIDTH: float = 0.10  # ±10% range
DEFAULT_SWAP_FEE: float = 0.0030  # 30 bps
DEFAULT_PERP_TRADING_COST: float = 0.0006  # 6 bps per trade
DEFAULT_INITIAL_TVL: float = 1_000_000.0  # $1M initial TVL
REINVEST_RATE: float = 0.65  # 65% reinvestment
RESERVE_RATE: float = 0.35  # 35% to reserve
HEDGE_REBALANCE_PRICE_THRESHOLD: float = 0.035  # 3.5% price move
HEDGE_REBALANCE_TIME_HOURS: int = 24  # 24 hours


# === DATA STRUCTURES ===========================================================

@dataclass
class LPPosition:
    """LP position with edge-rebalance capability."""
    lower: float
    upper: float
    liquidity: float
    price: float

    def get_amounts(self) -> tuple[float, float]:
        """Get (amount0, amount1) at current price."""
        sqrtP = np.sqrt(self.price)
        sqrtPa = np.sqrt(self.lower)
        sqrtPb = np.sqrt(self.upper)

        if self.price <= self.lower:
            amount0 = self.liquidity * (np.sqrt(self.upper) - np.sqrt(self.lower)) / (sqrtPa * sqrtPb)
            amount1 = 0.0
        elif self.price >= self.upper:
            amount0 = 0.0
            amount1 = self.liquidity * (sqrtPb - sqrtPa)
        else:
            amount0 = self.liquidity * (sqrtPb - sqrtP) / (sqrtP * sqrtPb)
            amount1 = self.liquidity * (sqrtP - sqrtPa)

        return float(amount0), float(amount1)

    def get_value_usd(self) -> float:
        """Get total value in USD."""
        amount0, amount1 = self.get_amounts()
        return amount0 * self.price + amount1

    def get_delta(self) -> float:
        """Get ETH exposure (delta)."""
        amount0, _ = self.get_amounts()
        return float(amount0)

    def update_price(self, new_price: float) -> None:
        """Update current price."""
        self.price = new_price

    def rebalance_to_new_range(self, new_lower: float, new_upper: float, new_price: float) -> None:
        """
        Rebalance LP position to a new range, conserving USD value.
        Computes new liquidity based on current USD value.
        """
        # Get current USD value before rebalancing
        current_value = self.get_value_usd()
        
        # Compute new liquidity needed for this value in the new range
        # We need to solve: value = amount0 * price + amount1
        # where amount0 and amount1 depend on liquidity L and the new range
        
        sqrtPa = np.sqrt(new_lower)
        sqrtPb = np.sqrt(new_upper)
        sqrtP = np.sqrt(new_price)
        
        # If price is in the new range, position is mixed
        if new_lower <= new_price <= new_upper:
            # Mixed position: value = L * (sqrtPb - sqrtP) / (sqrtP * sqrtPb) * price + L * (sqrtP - sqrtPa)
            # Simplify: value = L * [ (sqrtPb - sqrtP) * price / (sqrtP * sqrtPb) + (sqrtP - sqrtPa) ]
            # value = L * [ (sqrtPb - sqrtP) / sqrtPb + (sqrtP - sqrtPa) ]
            # value = L * [ sqrtPb/sqrtPb - sqrtP/sqrtPb + sqrtP - sqrtPa ]
            # value = L * [ 1 - sqrtP/sqrtPb + sqrtP - sqrtPa ]
            # Actually, let's use the direct formula:
            # amount0 = L * (sqrtPb - sqrtP) / (sqrtP * sqrtPb)
            # amount1 = L * (sqrtP - sqrtPa)
            # value = L * [ (sqrtPb - sqrtP) * price / (sqrtP * sqrtPb) + (sqrtP - sqrtPa) ]
            term = (sqrtPb - sqrtP) * new_price / (sqrtP * sqrtPb) + (sqrtP - sqrtPa)
            new_liquidity = current_value / term if term > 0 else current_value / (sqrtPb - sqrtPa)
        elif new_price < new_lower:
            # Fully in token0 (ETH): value = L * (sqrtPb - sqrtPa) / (sqrtPa * sqrtPb) * price
            new_liquidity = current_value * (sqrtPa * sqrtPb) / ((sqrtPb - sqrtPa) * new_price)
        else:  # new_price > new_upper
            # Fully in token1 (USDC): value = L * (sqrtPb - sqrtPa)
            new_liquidity = current_value / (sqrtPb - sqrtPa)
        
        # Update position
        self.lower = new_lower
        self.upper = new_upper
        self.liquidity = float(new_liquidity)
        self.price = new_price

    def edge_rebalance_up(self, new_price: float, width: float) -> None:
        """
        Edge-rebalance when price exits upper bound.
        New range: [old_upper, old_upper * (1 + width)]
        """
        old_upper = self.upper
        new_lower = old_upper
        new_upper = old_upper * (1.0 + width)
        self.rebalance_to_new_range(new_lower, new_upper, new_price)

    def edge_rebalance_down(self, new_price: float, width: float) -> None:
        """
        Edge-rebalance when price exits lower bound.
        New range: [old_lower * (1 - width), old_lower]
        """
        old_lower = self.lower
        new_upper = old_lower
        new_lower = old_lower * (1.0 - width)
        self.rebalance_to_new_range(new_lower, new_upper, new_price)
    
    def add_liquidity(self, extra_usd: float) -> None:
        """
        Add extra USD value to the LP position by increasing liquidity.
        Accounts for current position state (in range, above, or below).
        """
        if extra_usd <= 0:
            return
        
        sqrtP = np.sqrt(self.price)
        sqrtPa = np.sqrt(self.lower)
        sqrtPb = np.sqrt(self.upper)
        
        # Calculate how much liquidity is needed to add extra_usd value
        # This depends on where price is relative to the range
        if self.price <= self.lower:
            # Fully in token0 (ETH): value = L * (sqrtPb - sqrtPa) / (sqrtPa * sqrtPb) * price
            # So: L = value * (sqrtPa * sqrtPb) / ((sqrtPb - sqrtPa) * price)
            extra_liquidity = extra_usd * (sqrtPa * sqrtPb) / ((sqrtPb - sqrtPa) * self.price)
        elif self.price >= self.upper:
            # Fully in token1 (USDC): value = L * (sqrtPb - sqrtPa)
            # So: L = value / (sqrtPb - sqrtPa)
            extra_liquidity = extra_usd / (sqrtPb - sqrtPa)
        else:
            # Mixed position: value = amount0 * price + amount1
            # where amount0 = L * (sqrtPb - sqrtP) / (sqrtP * sqrtPb)
            # and amount1 = L * (sqrtP - sqrtPa)
            # So: value = L * [(sqrtPb - sqrtP) * price / (sqrtP * sqrtPb) + (sqrtP - sqrtPa)]
            term = (sqrtPb - sqrtP) * self.price / (sqrtP * sqrtPb) + (sqrtP - sqrtPa)
            extra_liquidity = extra_usd / term if term > 0 else extra_usd / (sqrtPb - sqrtPa)
        
        self.liquidity += float(extra_liquidity)


@dataclass
class PerpHedge:
    """Perpetual futures hedge with 2x leverage (capital efficient)."""
    size: float  # Short position size in ETH
    entry_price: float
    current_price: float
    cumulative_funding: float = 0.0
    cumulative_trading_costs: float = 0.0

    def update_price(self, new_price: float) -> None:
        self.current_price = new_price

    def get_pnl(self) -> float:
        """Unrealized P&L."""
        return (self.current_price - self.entry_price) * self.size

    def get_total_pnl(self) -> float:
        """Total P&L including funding and costs."""
        return self.get_pnl() + self.cumulative_funding - self.cumulative_trading_costs

    def add_funding(self, funding_rate: float, notional: float) -> None:
        """
        Add funding payment (positive = receive, negative = pay).
        
        Args:
            funding_rate: Funding rate (per hour, typically very small, e.g., 0.0001 = 0.01%)
            notional: Notional value of position (already = |size| * price)
        """
        self.cumulative_funding += funding_rate * notional

    def add_trading_cost(self, trade_size: float, cost_bps: float) -> None:
        cost = cost_bps * trade_size * self.current_price
        self.cumulative_trading_costs += cost


@dataclass
class Phase3Result:
    """Results from Phase 3 backtest."""
    # Configuration
    range_width: float
    initial_tvl: float
    final_tvl: float
    
    # Performance
    final_lp_value: float
    final_hedge_pnl: float
    total_fees_earned: float
    total_emissions_earned: float
    total_reinvested: float
    total_reserve: float
    total_trading_costs: float
    total_funding: float
    
    # Metrics
    net_profit: float
    net_return_pct: float
    net_apy: float
    
    # Time series
    lp_values: list[float] = field(default_factory=list)
    hedge_pnls: list[float] = field(default_factory=list)
    cumulative_profits: list[float] = field(default_factory=list)
    tvl_over_time: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"Phase3Result(\n"
            f"  range_width={self.range_width:.3f},\n"
            f"  final_tvl=${self.final_tvl:,.2f}, "
            f"hedge_pnl=${self.final_hedge_pnl:,.2f},\n"
            f"  fees=${self.total_fees_earned:,.2f}, "
            f"emissions=${self.total_emissions_earned:,.2f},\n"
            f"  reinvested=${self.total_reinvested:,.2f}, "
            f"reserve=${self.total_reserve:,.2f},\n"
            f"  net_profit=${self.net_profit:,.2f} "
            f"({self.net_return_pct:.2f}%), "
            f"APY={self.net_apy:.2f}%\n"
            f")"
        )


# === MAIN SIMULATION ===========================================================

def simulate_phase3_strategy(
    prices: np.ndarray,
    timestamps: np.ndarray,
    tvl_df: pd.DataFrame,
    fees_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
    volume_df: pd.DataFrame | None = None,  # Optional: not currently used but kept for future use
    range_width: float = DEFAULT_RANGE_WIDTH,
    initial_tvl: float = DEFAULT_INITIAL_TVL,
    swap_fee: float = DEFAULT_SWAP_FEE,
    perp_cost_bps: float = DEFAULT_PERP_TRADING_COST,
    reinvest_rate: float = REINVEST_RATE,
    hedge_price_threshold: float = HEDGE_REBALANCE_PRICE_THRESHOLD,
    hedge_time_hours: int = HEDGE_REBALANCE_TIME_HOURS,
    funding_rate: float = 0.0,
    store_series: bool = False,
) -> Phase3Result:
    """
    Simulate Phase 3 automated LP strategy with:
    - Edge-rebalance system
    - 24hr or 3.5% price move hedge rebalancing
    - 65% reinvestment of fees + emissions
    - Real Dune data for fees/TVL/volume
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 price points")
    
    # Initialize LP position
    P0 = float(prices[0])
    k = 1.0 + range_width
    lower0 = P0 / k
    upper0 = P0 * k
    initial_liquidity = initial_tvl / (np.sqrt(upper0) - np.sqrt(lower0))
    
    lp = LPPosition(
        lower=lower0,
        upper=upper0,
        liquidity=float(initial_liquidity),
        price=P0,
    )
    
    # Initialize hedge
    hedge = PerpHedge(
        size=0.0,
        entry_price=P0,
        current_price=P0,
    )
    
    # Prepare Dune data - create lookup for daily values
    def create_daily_lookup(df: pd.DataFrame, value_col: str) -> dict:
        """Create a lookup dict from date to daily value."""
        if df.empty:
            return {}
        
        lookup = {}
        for _, row in df.iterrows():
            day = pd.Timestamp(row['day']).date()
            lookup[day] = float(row[value_col])
        return lookup
    
    daily_fees_lookup = create_daily_lookup(fees_df, 'fees_usd')
    daily_tvl_lookup = create_daily_lookup(tvl_df, 'tvl_usd')
    
    # Process epoch data (weekly emissions)
    # Create a mapping from day number to daily emission value
    epoch_daily_emissions = {}
    if not epoch_df.empty:
        # Sort by epoch number
        epoch_df_sorted = epoch_df.sort_values('epoch').reset_index(drop=True)
        first_epoch = int(epoch_df_sorted.iloc[0]['epoch'])
        
        for _, row in epoch_df_sorted.iterrows():
            epoch_num = int(row['epoch'])
            emission_value = float(row.get('emission_value_usd', 0))
            # Each epoch is ~1 week, distribute evenly across 7 days
            daily_emission = emission_value / 7.0
            
            # Map to week number (relative to first epoch)
            week_num = epoch_num - first_epoch
            
            # Store daily emission for each day in this week
            for day in range(7):
                day_key = week_num * 7 + day
                epoch_daily_emissions[day_key] = daily_emission
    
    # Trackers
    total_fees = 0.0
    total_emissions = 0.0
    total_reinvested = 0.0
    total_reserve = 0.0
    cumulative_hedge_pnl = 0.0
    current_tvl = initial_tvl
    
    # Hedge rebalancing tracking
    last_hedge_rebalance_idx = 0
    last_hedge_price = P0
    
    # Time series
    lp_values_series = [initial_tvl] if store_series else []
    hedge_pnls_series = [0.0] if store_series else []
    cumulative_profits_series = [0.0] if store_series else []
    tvl_series = [initial_tvl] if store_series else []
    
    # Main simulation loop (hourly steps)
    for i in range(1, len(prices)):
        P_prev = float(prices[i - 1])
        P_curr = float(prices[i])
        ts_curr = timestamps[i]
        
        # Update LP price
        lp.update_price(P_curr)
        
        # Check for edge-rebalance (price exited range)
        if P_curr > lp.upper:
            # Price exited upper bound - edge-rebalance up
            lp.edge_rebalance_up(P_curr, range_width)
        elif P_curr < lp.lower:
            # Price exited lower bound - edge-rebalance down
            lp.edge_rebalance_down(P_curr, range_width)
        
        # Get daily fees from Dune data
        # Fees are daily totals for the entire pool, so we need to:
        # 1. Get the daily fee for this day
        # 2. Scale by our LP's share of TVL (use initial TVL as baseline)
        # 3. Divide by 24 to get hourly
        ts_date = pd.Timestamp(ts_curr).date()
        daily_pool_fees = daily_fees_lookup.get(ts_date, 0.0)
        daily_pool_tvl = daily_tvl_lookup.get(ts_date, initial_tvl)  # Use initial_tvl as fallback
        
        # Scale fees by our LP's share of TVL
        # Use a conservative approach: base share on initial TVL, allow modest growth
        if daily_pool_tvl > 0:
            # Base share: our initial investment relative to pool TVL
            base_share = min(initial_tvl / daily_pool_tvl, 1.0) if daily_pool_tvl >= initial_tvl else 1.0
            # Allow growth through reinvestment, but cap at 1.5x initial share or 100%
            # This prevents unrealistic feedback loops
            max_share = min(base_share * 1.5, 1.0)
            # Current share: our actual TVL (including reinvestment) relative to pool
            # But cap it to prevent over-scaling
            raw_share = current_tvl / daily_pool_tvl if daily_pool_tvl > 0 else 0.0
            current_share = min(raw_share, max_share)
        else:
            current_share = 0.0
        
        daily_lp_fees = daily_pool_fees * current_share
        hourly_fees = daily_lp_fees / 24.0
        total_fees += hourly_fees
        
        # Get emissions (weekly epochs)
        # Emissions are weekly totals for the entire pool, so we need to scale by LP share
        if isinstance(ts_curr, pd.Timestamp):
            days_since_start = (ts_curr - pd.Timestamp(timestamps[0])).total_seconds() / (24 * 3600)
        else:
            # numpy datetime64
            delta = (ts_curr - timestamps[0]).astype('timedelta64[D]').astype(float)
            days_since_start = delta
        day_num = int(days_since_start)
        daily_pool_emission = epoch_daily_emissions.get(day_num, 0.0)
        
        # Scale emissions by our LP's share of TVL (same logic as fees)
        if daily_pool_tvl > 0:
            base_share = initial_tvl / daily_pool_tvl if daily_pool_tvl >= initial_tvl else 1.0
            max_share = min(base_share * 2.0, 1.0)
            current_share = min(current_tvl / daily_pool_tvl, max_share) if daily_pool_tvl > 0 else 0.0
        else:
            current_share = 0.0
        
        daily_lp_emission = daily_pool_emission * current_share
        hourly_emission = daily_lp_emission / 24.0
        total_emissions += hourly_emission
        
        # Total yield this hour
        total_yield = hourly_fees + hourly_emission
        
        # Reinvest 65%, reserve 35%
        reinvest_amount = total_yield * reinvest_rate
        reserve_amount = total_yield * (1.0 - reinvest_rate)
        
        total_reinvested += reinvest_amount
        total_reserve += reserve_amount
        
        # Actually add reinvested amount to LP position (compounding)
        if reinvest_amount > 0:
            lp.add_liquidity(reinvest_amount)
        
        # Get current LP value AFTER reinvestment (for accurate time series)
        lp_value = lp.get_value_usd()
        current_tvl = lp_value  # Update TVL from actual LP value
        
        # Update hedge price
        hedge.update_price(P_curr)
        
        # Check if hedge needs rebalancing
        # Condition: 24 hours passed OR price moved ±3.5%
        hours_since_rebalance = (i - last_hedge_rebalance_idx)
        price_change_pct = abs(P_curr / last_hedge_price - 1.0)
        
        should_rebalance = (
            hours_since_rebalance >= hedge_time_hours or
            price_change_pct >= hedge_price_threshold
        )
        
        if should_rebalance or i == len(prices) - 1:
            # Realize P&L from current hedge
            if abs(hedge.size) > 1e-6:
                realized_pnl = hedge.get_pnl()
                cumulative_hedge_pnl += realized_pnl
            
            # Calculate required hedge size (match ETH exposure)
            target_delta = lp.get_delta()
            target_hedge_size = -target_delta  # Short to hedge
            
            # Calculate trade size
            trade_size = abs(target_hedge_size - hedge.size)
            
            if trade_size > 1e-6:
                hedge.add_trading_cost(trade_size, perp_cost_bps)
            
            # Update hedge position
            hedge.size = target_hedge_size
            hedge.entry_price = P_curr
            last_hedge_rebalance_idx = i
            last_hedge_price = P_curr
            
            # Add funding
            if abs(hedge.size) > 1e-6:
                notional = abs(hedge.size) * P_curr
                hedge.add_funding(funding_rate, notional)
        
        # Store time series
        if store_series:
            lp_values_series.append(lp_value)
            hedge_pnls_series.append(hedge.get_pnl())
            # Cumulative profit using same formula as final net_profit for consistency
            # Formula: LP value + reserve + hedge P&L + funding - initial - costs
            cumulative_profit = (
                lp_value +  # LP value (includes reinvested yield)
                total_reserve +  # Reserve capital (non-reinvested yield)
                hedge.get_pnl() +  # Current unrealized hedge P&L
                cumulative_hedge_pnl +  # Realized hedge P&L from rebalances
                hedge.cumulative_funding -  # Funding payments
                initial_tvl -  # Initial capital
                hedge.cumulative_trading_costs  # Trading costs
            )
            cumulative_profits_series.append(cumulative_profit)
            tvl_series.append(current_tvl)
    
    # Final calculations
    final_lp_value = lp.get_value_usd()
    final_unrealized_pnl = hedge.get_pnl() if abs(hedge.size) > 1e-6 else 0.0
    total_hedge_pnl = cumulative_hedge_pnl + final_unrealized_pnl
    
    # Net profit with compounding LP model:
    # final_lp_value already includes reinvested fees/emissions
    # total_reserve is the portion not reinvested
    # So: final LP + reserve + hedge P&L + funding - initial capital - costs
    net_profit = (
        final_lp_value +  # LP value (includes reinvested yield)
        total_reserve +  # Reserve capital (non-reinvested yield)
        total_hedge_pnl +  # Hedge P&L (price only)
        hedge.cumulative_funding -  # Funding payments
        initial_tvl -  # Initial capital
        hedge.cumulative_trading_costs  # Trading costs
    )
    
    net_return_pct = (net_profit / initial_tvl) * 100.0
    
    # Calculate APY (annualized)
    num_hours = len(prices) - 1
    num_years = num_hours / (365.25 * 24)
    if num_years > 0:
        net_apy = ((1.0 + net_profit / initial_tvl) ** (1.0 / num_years) - 1.0) * 100.0
    else:
        net_apy = 0.0
    
    # Update final_tvl to actual LP value (may differ from current_tvl due to compounding)
    final_tvl = final_lp_value
    
    return Phase3Result(
        range_width=range_width,
        initial_tvl=initial_tvl,
        final_tvl=final_tvl,
        final_lp_value=final_lp_value,
        final_hedge_pnl=total_hedge_pnl,
        total_fees_earned=total_fees,
        total_emissions_earned=total_emissions,
        total_reinvested=total_reinvested,
        total_reserve=total_reserve,
        total_trading_costs=hedge.cumulative_trading_costs,
        total_funding=hedge.cumulative_funding,
        net_profit=net_profit,
        net_return_pct=net_return_pct,
        net_apy=net_apy,
        lp_values=lp_values_series if store_series else [],
        hedge_pnls=hedge_pnls_series if store_series else [],
        cumulative_profits=cumulative_profits_series if store_series else [],
        tvl_over_time=tvl_series if store_series else [],
    )

