import os
import zipfile
import pandas as pd

# Base directory = project root (one level above src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "binance_raw")
OUT_FILE = os.path.join(BASE_DIR, "data", "eth_1min.csv")


def load_all_binance_1m():
    if not os.path.isdir(RAW_PATH):
        raise FileNotFoundError(f"RAW_PATH does not exist: {RAW_PATH}")

    frames = []

    for fname in sorted(os.listdir(RAW_PATH)):
        if not fname.endswith(".zip"):
            continue

        full_path = os.path.join(RAW_PATH, fname)
        print(f"Loading {fname} ...")

        with zipfile.ZipFile(full_path, "r") as z:
            inner_name = z.namelist()[0]  # the CSV inside the zip
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

                # Ensure open_time is numeric; coerce bad values to NaN
                df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
                df = df.dropna(subset=["open_time"])

                # Convert ms → datetime; coerce invalid to NaT and drop them
                df["timestamp"] = pd.to_datetime(
                    df["open_time"],
                    unit="ms",
                    errors="coerce",
                )
                df = df.dropna(subset=["timestamp"])

                # Keep only what we actually need
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                frames.append(df)

    if not frames:
        raise RuntimeError(f"No valid .zip files found in {RAW_PATH}")

    final = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)
    print("Final dataframe size:", len(final))

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    final.to_csv(OUT_FILE, index=False)
    print(f"Saved merged file to {OUT_FILE}")


if __name__ == "__main__":
    load_all_binance_1m()
