#!/usr/bin/env python3
"""Main script for delta-neutral LP backtesting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from lp_engine import (
    optimize_delta_neutral_lp,
    simulate_delta_neutral_lp,
    DEFAULT_RANGE_WIDTH,
    DEFAULT_REBALANCE_FREQ,
)
from lp_simulator import CSV_PATH as DEFAULT_CSV_PATH, _auto_pick_price_column


def load_price_data(
    csv_path: Path = DEFAULT_CSV_PATH,
    price_column: str | None = None,
    downsample_step: int = 1,  # Use all data for delta-neutral (more accurate)
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load price data and return (prices, timestamps).
    For delta-neutral, we want higher frequency data.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Price file not found at {csv_path}.\n"
            "→ Make sure data/eth_1min.csv exists."
        )
    
    df = pd.read_csv(csv_path)
    col = _auto_pick_price_column(df, price_column)
    
    prices = df[col].astype(float).to_numpy()
    
    # Get timestamps if available
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"]).to_numpy()
    else:
        timestamps = None
    
    # Downsample if requested
    if downsample_step > 1:
        prices = prices[::downsample_step]
        if timestamps is not None:
            timestamps = timestamps[::downsample_step]
    
    if prices.size < 2:
        raise ValueError("Price series is too short after loading/downsampling.")
    
    return prices, timestamps


def plot_results(
    prices: np.ndarray,
    timestamps: np.ndarray | None,
    result,
    output_file: str | None = None,
) -> None:
    """Plot comprehensive visualization of backtest results."""
    if not result.lp_values:
        print("Warning: No time series data to plot. Run with store_series=True")
        return
    
    # Create figure with subplots - increased size and better spacing
    fig = plt.figure(figsize=(18, 14))
    # Adjust grid layout: more space between subplots
    gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.4, 
                          left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    # Prepare time axis
    if timestamps is not None and len(timestamps) == len(prices):
        time_axis = timestamps[:len(result.lp_values)]
        use_dates = True
    else:
        time_axis = np.arange(len(result.lp_values))
        use_dates = False
    
    # 1. Price with LP range bounds (top left)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, prices[:len(result.lp_values)], 'b-', alpha=0.7, linewidth=1, label='ETH Price')
    
    # Calculate range bounds over time (simplified - would need to track in simulation)
    # For now, show a band around price
    k = 1.0 + result.range_width
    upper_band = prices[:len(result.lp_values)] * k
    lower_band = prices[:len(result.lp_values)] / k
    ax1.fill_between(time_axis, lower_band, upper_band, alpha=0.2, color='green', label=f'LP Range (±{result.range_width*100:.0f}%)')
    
    ax1.set_ylabel('Price (USD)', fontsize=11)
    ax1.set_title('ETH Price with LP Range', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    if use_dates:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax1.set_xlabel('Time Step', fontsize=10)
    
    # 2. LP Value over time (top right)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_axis, result.lp_values, 'g-', linewidth=2, label='LP Value')
    ax2.axhline(y=result.initial_tvl, color='r', linestyle='--', alpha=0.5, label='Initial TVL')
    ax2.set_ylabel('Value (USD)', fontsize=11)
    ax2.set_title('LP Position Value', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    if use_dates:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax2.set_xlabel('Time Step', fontsize=10)
    
    # 3. Hedge P&L over time (top right, second row)
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
    else:
        ax3.set_xlabel('Time Step', fontsize=10)
    
    # 4. Cumulative Profit (bottom left)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(time_axis, result.cumulative_profits, 'purple', linewidth=2, label='Cumulative Profit')
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_ylabel('Profit (USD)', fontsize=11)
    ax4.set_xlabel('Time', fontsize=11)
    ax4.set_title(f'Cumulative Net Profit (Final: ${result.net_profit:,.2f}, APY: {result.net_apy:.2f}%)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax4.legend(framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    if use_dates:
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax4.set_xlabel('Time Step', fontsize=11)
    
    # 5. Summary metrics (bottom right)
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    metrics_text = f"""
    DELTA-NEUTRAL LP BACKTEST RESULTS
    
    Configuration:
    • Range Width: ±{result.range_width*100:.1f}%
    • Rebalance Frequency: {result.rebalance_freq} minutes
    • Initial TVL: ${result.initial_tvl:,.2f}
    
    Final Metrics:
    • Final LP Value: ${result.final_lp_value:,.2f}
    • Hedge P&L: ${result.final_hedge_pnl:,.2f}
    • Fees Earned: ${result.total_fees_earned:,.2f}
    • Arbitrage Losses: ${result.total_arbitrage_loss:,.2f}
    • Trading Costs: ${result.total_trading_costs:,.2f}
    
    Net Results:
    • Net Profit: ${result.net_profit:,.2f}
    • Net Return: {result.net_return_pct:.2f}%
    • Net APY: {result.net_apy:.2f}%
    """
    
    ax5.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add main title with proper spacing
    plt.suptitle('Delta-Neutral LP Backtest Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot if filename provided
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    
    # Always show the plot (user can close the window)
    plt.show()


def run_single_backtest(
    csv_path: Path = DEFAULT_CSV_PATH,
    range_width: float = DEFAULT_RANGE_WIDTH,
    rebalance_freq: int = DEFAULT_REBALANCE_FREQ,
    downsample_step: int = 1,
    store_series: bool = False,
    plot: bool = False,
    plot_file: str | None = None,
) -> None:
    """Run a single delta-neutral LP backtest with specified parameters."""
    print("=" * 70)
    print("Delta-Neutral LP Backtest")
    print("=" * 70)
    print(f"Loading price data from {csv_path.name}...")
    
    prices, timestamps = load_price_data(csv_path, downsample_step=downsample_step)
    
    print(f"Loaded {len(prices):,} price points")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"Time period: {len(prices) / (365.25 * 24 * 60):.2f} years (assuming 1-min data)")
    print()
    
    print("Configuration:")
    print(f"  Range width: ±{range_width*100:.1f}%")
    print(f"  Rebalance frequency: {rebalance_freq} minutes")
    print(f"  Swap fee: 30 bps")
    print(f"  Perp trading cost: 6 bps")
    print(f"  Initial TVL: $1,000,000")
    print()
    
    print("Running simulation...")
    # Enable series storage if plotting
    result = simulate_delta_neutral_lp(
        prices=prices,
        timestamps=timestamps,
        range_width=range_width,
        rebalance_freq=rebalance_freq,
        store_series=store_series or plot,  # Auto-enable if plotting
    )
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(result)
    print()
    
    print("Breakdown:")
    print(f"  Initial TVL:        ${result.initial_tvl:,.2f}")
    print(f"  Final LP value:      ${result.final_lp_value:,.2f}")
    print(f"  LP value change:     ${result.final_lp_value - result.initial_tvl:,.2f}")
    print(f"  Hedge P&L:           ${result.final_hedge_pnl:,.2f}")
    print(f"  Fees earned:         ${result.total_fees_earned:,.2f}")
    print(f"  Arbitrage losses:    ${result.total_arbitrage_loss:,.2f}")
    print(f"  Trading costs:      ${result.total_trading_costs:,.2f}")
    print(f"  Funding payments:   ${result.total_funding:,.2f}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  NET PROFIT:         ${result.net_profit:,.2f}")
    print(f"  NET RETURN:         {result.net_return_pct:.2f}%")
    print(f"  NET APY:            {result.net_apy:.2f}%")
    print()
    
    # Plot if requested
    if plot:
        print("Generating plots...")
        # If plot requested but no filename, use default
        if plot_file is None:
            plot_file = "backtest_results.png"
        plot_results(prices, timestamps, result, output_file=plot_file)
        print()


def run_optimization(
    csv_path: Path = DEFAULT_CSV_PATH,
    downsample_step: int = 5,  # Downsample for faster optimization
) -> None:
    """Run optimization over range widths and rebalance frequencies."""
    print("=" * 70)
    print("Delta-Neutral LP Optimization")
    print("=" * 70)
    print(f"Loading price data from {csv_path.name}...")
    
    prices, _ = load_price_data(csv_path, downsample_step=downsample_step)
    
    print(f"Loaded {len(prices):,} price points (downsampled by {downsample_step})")
    print()
    
    print("Running grid search...")
    print("Testing combinations of:")
    print("  Range widths: 5%, 10%, 15%, 20%")
    print("  Rebalance frequencies: 5min, 15min, 30min, 60min")
    print()
    
    best, all_results = optimize_delta_neutral_lp(
        prices=prices,
        range_widths=(0.05, 0.10, 0.15, 0.20),
        rebalance_freqs=(5, 15, 30, 60),
    )
    
    print()
    print("=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print()
    print("Top 10 configurations by APY:")
    print()
    
    sorted_results = sorted(all_results, key=lambda r: r.net_apy, reverse=True)
    for i, r in enumerate(sorted_results[:10], 1):
        print(
            f"{i:2d}. width=±{r.range_width*100:4.0f}%, "
            f"rebalance={r.rebalance_freq:3d}min → "
            f"APY={r.net_apy:6.2f}%, "
            f"profit=${r.net_profit:10,.2f}, "
            f"fees=${r.total_fees_earned:10,.2f}"
        )
    
    print()
    print("=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(best)
    print()


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Delta-Neutral LP Backtesting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single backtest
  python delta_neutral_backtest.py
  
  # Run with plotting
  python delta_neutral_backtest.py --plot
  
  # Run optimization
  python delta_neutral_backtest.py optimize
  
  # Run with custom parameters and save plot
  python delta_neutral_backtest.py --range-width 0.15 --rebalance-freq 30 --plot --plot-file results.png
        """
    )
    
    parser.add_argument('command', nargs='?', default='backtest',
                       help='Command: "backtest" (default) or "optimize"')
    parser.add_argument('--range-width', type=float, default=DEFAULT_RANGE_WIDTH,
                       help=f'Range width (default: {DEFAULT_RANGE_WIDTH})')
    parser.add_argument('--rebalance-freq', type=int, default=DEFAULT_REBALANCE_FREQ,
                       help=f'Rebalance frequency in minutes (default: {DEFAULT_REBALANCE_FREQ})')
    parser.add_argument('--downsample', type=int, default=1,
                       help='Downsample step (default: 1, use 5-10 for faster runs)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    parser.add_argument('--plot-file', type=str, default=None,
                       help='Save plot to file (e.g., results.png). If --plot is used without --plot-file, saves to backtest_results.png')
    
    args = parser.parse_args()
    
    if args.command == 'optimize':
        # Run optimization
        run_optimization()
    else:
        # Run single backtest
        run_single_backtest(
            range_width=args.range_width,
            rebalance_freq=args.rebalance_freq,
            downsample_step=args.downsample,
            plot=args.plot,
            plot_file=args.plot_file,
        )

