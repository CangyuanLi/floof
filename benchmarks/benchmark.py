# Imports
from pathlib import Path

import cutils
import pandas as pd

import floof

DAT_PATH = Path(__file__).resolve().parents[0] / "data"

def benchmark_dictionary():
    with open("/usr/share/dict/words", "rt") as f:
        dct_words = f.readlines()

    idx = len(dct_words) // 2

    half1 = dct_words[:idx]
    half2 = dct_words[idx:]

    df1 = pd.DataFrame({"name1": half1})
    df2 = pd.DataFrame({"name2": half2})
    df1 = df1.sample(2_000)
    df2 = df2.sample(2_000)

    matcher = floof.Matcher(df1["name1"], df2["name2"])

    cutils.time_func(lambda: matcher.damerau_levenshtein(ncpus=1), warmups=1, iterations=2)
    cutils.time_func(lambda: matcher.damerau_levenshtein_trie(ncpus=1), warmups=1, iterations=2)

def benchmark_real_data():
    df = pd.read_csv(DAT_PATH / "test_data.csv")
    df = df.sample(2_000)

    matcher = floof.Matcher(df["name1"], df["name2"])

    cutils.time_func(lambda: matcher.damerau_levenshtein(ncpus=1), warmups=1, iterations=2)
    cutils.time_func(lambda: matcher.damerau_levenshtein_trie(ncpus=1), warmups=1, iterations=2)

def main():
    benchmark_dictionary()


if __name__ == "__main__":
    main()