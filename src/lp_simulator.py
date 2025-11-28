import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


# === CONFIG YOU MAY NEED TO CHANGE ONCE ===
# Path is relative to the project root (directory that contains "src")
CSV_PATH = "data/eth_1min.csv"      # e.g. "data/btc_hourly.csv"
PRICE_COLUMN = "close"            # e.g. "close", "Price", etc.
DEFAULT_BAND_WIDTH_PCT = 0.5      # ±50% around initial price
# ==========================================


@dataclass
class LPSimulationResult:
    lower: float
    upper: float
    initial_price: float
    pnl: float
    fees_earned: float
    # add extra fields as needed


def get_project_root() -> Path:
    """
    Assume lp_simulator.py lives in src/, so project root is .. from here.
    """
    return Path(__file__).resolve().parent.parent


def get_price_series(
    csv_path: str = CSV_PATH,
    price_column: str = PRICE_COLUMN,
) -> np.ndarray:
    """
    Load price series from a CSV in the project.

    Returns
    -------
    np.ndarray of float
    """
    root = get_project_root()
    full_path = root / csv_path

    if not full_path.exists():
        raise FileNotFoundError(
            f"Price file not found at {full_path}.\n"
            f"→ Fix CSV_PATH in lp_simulator.py or create the file there."
        )

    df = pd.read_csv(full_path)

    if price_column not in df.columns:
        raise KeyError(
            f"Column '{price_column}' not found in {full_path}.\n"
            f"Available columns: {list(df.columns)}\n"
            f"→ Fix PRICE_COLUMN in lp_simulator.py."
        )

    series = df[price_column].astype(float).to_numpy()

    if series.size == 0:
        raise ValueError(f"Price column '{price_column}' in {full_path} is empty.")

    return series


def simulate_static_lp(
    price_series: np.ndarray | None = None,
    band_width_pct: float = DEFAULT_BAND_WIDTH_PCT,
) -> LPSimulationResult:
    """
    Simulate a static LP position with an automatically chosen price band.

    Parameters
    ----------
    price_series : array-like, optional
        Sequence of prices. If None, loads from CSV_PATH / PRICE_COLUMN.
    band_width_pct : float, default DEFAULT_BAND_WIDTH_PCT
        Half-width of the band as a percentage of the initial price.
        Example: 0.5 => [initial_price * 0.5, initial_price * 1.5]

    Returns
    -------
    LPSimulationResult
    """

    # 1) Get prices
    if price_series is None:
        price_series = get_price_series()

    price_series = np.asarray(price_series, dtype=float)
    if price_series.size == 0:
        raise ValueError("Price series is empty after loading.")

    # 2) Initial price
    initial_price = float(price_series[0])

    # 3) Auto-set band around initial price (no manual inputs)
    if band_width_pct <= 0:
        raise ValueError("band_width_pct must be positive.")
    lower = initial_price * (1 - band_width_pct)
    upper = initial_price * (1 + band_width_pct)

    if not (lower <= initial_price <= upper):
        # This should never happen, but guard anyway
        raise ValueError(
            f"Initial price {initial_price:.2f} is not within "
            f"[{lower:.2f}, {upper:.2f}]. Check band_width_pct."
        )

    # 4) Your existing LP simulation logic goes here
    # ---------------------------------------------
    # TODO: replace this placeholder with your actual math.
    # For now, we just set them to 0 so the script runs.
    pnl = 0.0
    fees_earned = 0.0

    # Example: iterate over price_series and compute pnl, fees, etc.
    # for p in price_series[1:]:
    #     ...

    return LPSimulationResult(
        lower=lower,
        upper=upper,
        initial_price=initial_price,
        pnl=pnl,
        fees_earned=fees_earned,
    )


if __name__ == "__main__":
    result = simulate_static_lp()
    print(result)
