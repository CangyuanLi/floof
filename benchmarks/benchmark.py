# Imports
from pathlib import Path

import floof
import pandas as pd
import perfplot

DAT_PATH = Path(__file__).resolve().parents[0] / "data"

def data_setup(df: pd.DataFrame, n: int) -> floof.Matcher:
    df = df.sample(n)

    return floof.Matcher(df["name1"], df["name2"])

def main():
    df = pd.read_csv(DAT_PATH / "test_data.csv")

    perfplot.save(
        filename=Path(__file__).resolve().parents[0] / "edit_distance_scorers.png",
        setup=lambda n: data_setup(df, n),
        n_range=[2**k for k in range(15)],
        kernels=[
            lambda m: m.hamming(),
            lambda m: m.hamming(ascii_only=True),
            lambda m: m.jaro(),
            lambda m: m.jaro_winkler(),
            lambda m: m.levenshtein(),
            lambda m: m.damerau_levenshtein()
        ],
        time_unit="auto",
        labels=["hamming", "hamming (ascii)", "jaro", "jaro_winkler", "levenshtein", "damerau_levenshtein"],
        xlabel="Number of rows",
        equality_check=None,
    )


if __name__ == "__main__":
    main()
