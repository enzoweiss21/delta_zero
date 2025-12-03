#!/usr/bin/env python3
"""Phase 3 Backtest: Automated LP Strategy with Real Dune Data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from phase3_engine import simulate_phase3_strategy, DEFAULT_RANGE_WIDTH, DEFAULT_INITIAL_TVL
from load_phase3_data import load_dune_data

BASE_DIR = Path(__file__).resolve().parents[1]
PRICE_FILE = BASE_DIR / "data" / "eth_1h_phase3.csv"


def load_phase3_price_data(start_date: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load 1-hour ETH price data.
    
    Args:
        start_date: Optional start date filter (e.g., '2025-01-01'). 
                   If None, uses all available data.
    """
    if not PRICE_FILE.exists():
        raise FileNotFoundError(
            f"Price file not found: {PRICE_FILE}\n"
            "Run: python src/load_phase3_data.py first"
        )
    
    df = pd.read_csv(PRICE_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter by start date if provided
    if start_date:
        start_ts = pd.Timestamp(start_date)
        df = df[df['timestamp'] >= start_ts]
        if len(df) == 0:
            raise ValueError(f"No price data found after {start_date}")
        print(f"Filtered to start from {start_date}: {len(df):,} rows")
    
    prices = df['close'].astype(float).to_numpy()
    timestamps = df['timestamp'].to_numpy()
    
    return prices, timestamps


def plot_phase3_results(
    prices: np.ndarray,
    timestamps: np.ndarray,
    result,
    output_file: str | None = None,
) -> None:
    """Plot Phase 3 backtest results (same setup as Phase 2)."""
    if not result.lp_values:
        print("Warning: No time series data to plot. Run with store_series=True")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.4,
                          left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    # Prepare time axis
    time_axis = timestamps[:len(result.lp_values)]
    use_dates = True
    
    # 1. Price with LP range bounds (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, prices[:len(result.lp_values)], 'b-', alpha=0.7, linewidth=1, label='ETH Price')
    
    # Show range bounds (simplified - would need to track in simulation)
    k = 1.0 + result.range_width
    upper_band = prices[:len(result.lp_values)] * k
    lower_band = prices[:len(result.lp_values)] / k
    ax1.fill_between(time_axis, lower_band, upper_band, alpha=0.2, color='green',
                     label=f'LP Range (±{result.range_width*100:.0f}%)')
    
    ax1.set_ylabel('Price (USD)', fontsize=11)
    ax1.set_title('ETH Price with LP Range (Edge-Rebalanced)', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    if use_dates:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. LP Value over time (top left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_axis, result.lp_values, 'g-', linewidth=2, label='LP Value')
    if result.tvl_over_time:
        ax2.plot(time_axis, result.tvl_over_time, 'orange', linestyle='--', linewidth=1.5, label='TVL (with reinvestment)')
    ax2.axhline(y=result.initial_tvl, color='r', linestyle='--', alpha=0.5, label='Initial TVL')
    ax2.set_ylabel('Value (USD)', fontsize=11)
    ax2.set_title('LP Position Value', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    if use_dates:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Hedge P&L over time (top right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_axis, result.hedge_pnls, 'r-', linewidth=2, label='Hedge P&L')
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_ylabel('P&L (USD)', fontsize=11)
    ax3.set_title('Perpetual Hedge P&L', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    if use_dates:
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Cumulative Profit (bottom, full width)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(time_axis, result.cumulative_profits, 'purple', linewidth=2, label='Cumulative Profit')
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_ylabel('Profit (USD)', fontsize=11)
    ax4.set_xlabel('Time', fontsize=11)
    ax4.set_title(
        f'Cumulative Net Profit (Final: ${result.net_profit:,.2f}, APY: {result.net_apy:.2f}%)',
        fontsize=12, fontweight='bold', pad=10
    )
    ax4.legend(framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    if use_dates:
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Summary metrics (bottom, full width)
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    metrics_text = f"""
    PHASE 3: AUTOMATED LP STRATEGY BACKTEST RESULTS
    
    Configuration:
    • Range Width: ±{result.range_width*100:.1f}% (Edge-Rebalanced)
    • Hedge Rebalance: 24hr OR ±3.5% price move
    • Reinvestment Rate: 65% of fees + emissions
    • Initial TVL: ${result.initial_tvl:,.2f}
    
    Final Metrics:
    • Final LP Value:      ${result.final_lp_value:,.2f}
    • Final TVL:           ${result.final_tvl:,.2f}
    • Hedge P&L:           ${result.final_hedge_pnl:,.2f}
    • Fees Earned:         ${result.total_fees_earned:,.2f}
    • Emissions Earned:    ${result.total_emissions_earned:,.2f}
    • Total Reinvested:     ${result.total_reinvested:,.2f}
    • Reserve Capital:      ${result.total_reserve:,.2f}
    • Trading Costs:       ${result.total_trading_costs:,.2f}
    • Funding Payments:    ${result.total_funding:,.2f}
    
    Net Results:
    • Net Profit:          ${result.net_profit:,.2f}
    • Net Return:          {result.net_return_pct:.2f}%
    • Net APY:             {result.net_apy:.2f}%
    """
    
    ax5.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Phase 3: Automated LP Strategy Backtest Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    
    plt.show()


def run_phase3_backtest(
    range_width: float = DEFAULT_RANGE_WIDTH,
    initial_tvl: float = DEFAULT_INITIAL_TVL,
    plot: bool = False,
    plot_file: str | None = None,
    start_date: str | None = None,
) -> None:
    """
    Run Phase 3 backtest with real Dune data.
    
    Args:
        range_width: LP range width (default: ±10%)
        initial_tvl: Initial TVL in USD (default: $1M)
        plot: Whether to generate plots
        plot_file: Optional file path to save plot
        start_date: Optional start date (e.g., '2025-01-01') to filter data
    """
    print("=" * 70)
    print("Phase 3: Automated LP Strategy Backtest")
    if start_date:
        print(f"Starting from: {start_date}")
    print("=" * 70)
    
    # Load price data
    print("\nLoading 1-hour ETH price data...")
    prices, timestamps = load_phase3_price_data(start_date=start_date)
    print(f"Loaded {len(prices):,} price points")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"Date range: {pd.Timestamp(timestamps[0])} to {pd.Timestamp(timestamps[-1])}")
    
    # Load Dune data
    print("\nLoading Dune CSV data...")
    dune_data = load_dune_data()
    print(f"Volume data: {len(dune_data['volume'])} days")
    print(f"TVL data: {len(dune_data['tvl'])} days")
    print(f"Fees data: {len(dune_data['fees'])} days")
    print(f"Epoch data: {len(dune_data['epoch'])} epochs")
    
    print("\nConfiguration:")
    print(f"  Range width: ±{range_width*100:.1f}% (Edge-Rebalanced)")
    print(f"  Hedge rebalance: 24hr OR ±3.5% price move")
    print(f"  Reinvestment: 65% of fees + emissions")
    print(f"  Initial TVL: ${initial_tvl:,.2f}")
    
    print("\nRunning simulation...")
    result = simulate_phase3_strategy(
        prices=prices,
        timestamps=timestamps,
        tvl_df=dune_data['tvl'],
        fees_df=dune_data['fees'],
        epoch_df=dune_data['epoch'],
        volume_df=dune_data['volume'],  # Optional: not currently used
        range_width=range_width,
        initial_tvl=initial_tvl,
        store_series=plot or plot_file is not None,
    )
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(result)
    print()
    
    print("Breakdown:")
    print(f"  Initial TVL:        ${result.initial_tvl:,.2f}")
    print(f"  Final LP value:      ${result.final_lp_value:,.2f}")
    print(f"  Final TVL:           ${result.final_tvl:,.2f}")
    print(f"  LP value change:     ${result.final_lp_value - result.initial_tvl:,.2f}")
    print(f"  Hedge P&L:           ${result.final_hedge_pnl:,.2f}")
    print(f"  Fees earned:         ${result.total_fees_earned:,.2f}")
    print(f"  Emissions earned:    ${result.total_emissions_earned:,.2f}")
    print(f"  Total reinvested:   ${result.total_reinvested:,.2f}")
    print(f"  Reserve capital:     ${result.total_reserve:,.2f}")
    print(f"  Trading costs:      ${result.total_trading_costs:,.2f}")
    print(f"  Funding payments:   ${result.total_funding:,.2f}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  NET PROFIT:         ${result.net_profit:,.2f}")
    print(f"  NET RETURN:         {result.net_return_pct:.2f}%")
    print(f"  NET APY:            {result.net_apy:.2f}%")
    print()
    
    # Plot if requested
    if plot or plot_file:
        print("Generating plots...")
        if plot_file is None:
            plot_file = "phase3_backtest_results.png"
        plot_phase3_results(prices, timestamps, result, output_file=plot_file)
        print()


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 3: Automated LP Strategy Backtest')
    parser.add_argument('--range-width', type=float, default=DEFAULT_RANGE_WIDTH,
                       help=f'Range width (default: {DEFAULT_RANGE_WIDTH})')
    parser.add_argument('--initial-tvl', type=float, default=DEFAULT_INITIAL_TVL,
                       help=f'Initial TVL (default: ${DEFAULT_INITIAL_TVL:,.0f})')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--plot-file', type=str, default=None,
                       help='Save plot to file')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date filter (e.g., 2025-01-01)')
    
    args = parser.parse_args()
    
    run_phase3_backtest(
        range_width=args.range_width,
        initial_tvl=args.initial_tvl,
        plot=args.plot,
        plot_file=args.plot_file,
        start_date=args.start_date,
    )

