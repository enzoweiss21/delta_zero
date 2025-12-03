#!/usr/bin/env python3
"""Load Phase 3 data: 1-hour ETH prices and Dune CSV data."""

import os
import zipfile
import pandas as pd
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parents[1]
PHASE3_DATA_DIR = BASE_DIR / "data" / "phase_3_Data" / "Dune_Data"
PRICE_ZIP_DIR = PHASE3_DATA_DIR / "eth_price_binance"
OUT_PRICE_FILE = BASE_DIR / "data" / "eth_1h_phase3.csv"

# Pool address
POOL_ADDRESS = "0xb2cc224c1c9fee385f8ad6a55b4d94e92359dc59"


def load_1h_price_data():
    """Load and merge all 1-hour ETH price ZIP files."""
    if not PRICE_ZIP_DIR.exists():
        raise FileNotFoundError(f"Price ZIP directory not found: {PRICE_ZIP_DIR}")
    
    frames = []
    
    for zip_file in sorted(PRICE_ZIP_DIR.glob("ETHUSDT-1h-*.zip")):
        print(f"Loading {zip_file.name}...")
        
        with zipfile.ZipFile(zip_file, "r") as z:
            inner_name = z.namelist()[0]  # The CSV inside the zip
            with z.open(inner_name) as f:
                df = pd.read_csv(
                    f,
                    header=None,
                    names=[
                        "open_time", "open", "high", "low", "close",
                        "volume", "close_time", "quote_volume",
                        "num_trades", "taker_buy_base",
                        "taker_buy_quote", "ignore",
                    ],
                )
                
                # Convert timestamps
                df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
                df = df.dropna(subset=["open_time"])
                
                # Detect timestamp unit: microseconds have 16 digits, milliseconds have 13
                # 2024-05-01 in ms = 1714521600000 (13 digits)
                # 2025-01-01 in us = 1735689600000000 (16 digits)
                sample_ts = df["open_time"].iloc[0] if len(df) > 0 else 0
                ts_str = str(int(sample_ts))
                if len(ts_str) >= 16:
                    # Microseconds (16+ digits) - convert to milliseconds
                    df["open_time"] = df["open_time"] / 1000.0
                # Otherwise it's already in milliseconds (13 digits)
                
                df["timestamp"] = pd.to_datetime(
                    df["open_time"],
                    unit="ms",
                    errors="coerce",
                )
                df = df.dropna(subset=["timestamp"])
                
                # Keep only what we need
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                frames.append(df)
    
    if not frames:
        raise RuntimeError(f"No valid ZIP files found in {PRICE_ZIP_DIR}")
    
    final = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)
    
    # Filter to only valid dates (2024-05-01 onwards to match Dune data start)
    final = final[final['timestamp'] >= pd.Timestamp('2024-05-01')]
    # Also filter to end at 2025-10-15 to match Dune data end
    final = final[final['timestamp'] <= pd.Timestamp('2025-10-15 23:59:59')]
    
    print(f"Final price dataframe size: {len(final)}")
    print(f"Date range: {final['timestamp'].min()} to {final['timestamp'].max()}")
    
    os.makedirs(OUT_PRICE_FILE.parent, exist_ok=True)
    final.to_csv(OUT_PRICE_FILE, index=False)
    print(f"Saved merged price file to {OUT_PRICE_FILE}")
    
    return final


def load_dune_data():
    """Load all Dune CSV data files."""
    data = {}
    
    # Volume
    volume_file = PHASE3_DATA_DIR / f"Volume_Pool_ {POOL_ADDRESS} - Sheet1.csv"
    if volume_file.exists():
        df = pd.read_csv(volume_file)
        df["day"] = pd.to_datetime(df["day"])
        data["volume"] = df
        print(f"Loaded volume data: {len(df)} rows")
    else:
        print(f"Warning: Volume file not found: {volume_file}")
        data["volume"] = pd.DataFrame()
    
    # TVL
    tvl_file = PHASE3_DATA_DIR / f"TVL_Pool_{POOL_ADDRESS} - Sheet1.csv"
    if tvl_file.exists():
        df = pd.read_csv(tvl_file)
        df["day"] = pd.to_datetime(df["day"])
        data["tvl"] = df
        print(f"Loaded TVL data: {len(df)} rows")
    else:
        print(f"Warning: TVL file not found: {tvl_file}")
        data["tvl"] = pd.DataFrame()
    
    # Fees
    fees_file = PHASE3_DATA_DIR / f"Fees_Pool_{POOL_ADDRESS} - Sheet1.csv"
    if fees_file.exists():
        df = pd.read_csv(fees_file)
        df["day"] = pd.to_datetime(df["day"])
        data["fees"] = df
        print(f"Loaded fees data: {len(df)} rows")
    else:
        print(f"Warning: Fees file not found: {fees_file}")
        data["fees"] = pd.DataFrame()
    
    # Epoch (weekly emissions)
    epoch_file = PHASE3_DATA_DIR / f"Epoch_pool_{POOL_ADDRESS} - Sheet1.csv"
    if epoch_file.exists():
        df = pd.read_csv(epoch_file)
        data["epoch"] = df
        print(f"Loaded epoch data: {len(df)} rows")
    else:
        print(f"Warning: Epoch file not found: {epoch_file}")
        data["epoch"] = pd.DataFrame()
    
    return data


if __name__ == "__main__":
    print("Loading Phase 3 data...")
    print("=" * 70)
    
    # Load price data
    print("\n1. Loading 1-hour ETH price data...")
    price_df = load_1h_price_data()
    
    # Load Dune data
    print("\n2. Loading Dune CSV data...")
    dune_data = load_dune_data()
    
    print("\n" + "=" * 70)
    print("Phase 3 data loaded successfully!")
    print(f"Price data: {len(price_df)} rows")
    print(f"Volume data: {len(dune_data['volume'])} rows")
    print(f"TVL data: {len(dune_data['tvl'])} rows")
    print(f"Fees data: {len(dune_data['fees'])} rows")
    print(f"Epoch data: {len(dune_data['epoch'])} rows")

