# Imports
from pathlib import Path

import cutils
import floof
import pandas as pd

DAT_PATH = Path(__file__).resolve().parents[0] / "data"

def main():
    df = pd.read_csv(DAT_PATH / "test_data.csv")
    df = df.sample(10_000)

    matcher = floof.Matcher(df["name1"], df["name2"])
    cutils.time_func(matcher.hamming, warmups=1, iterations=10)
    cutils.time_func(matcher.hamming, warmups=1, iterations=10)

if __name__ == "__main__":
    main()