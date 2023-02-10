from pathlib import Path

import cutils
import pandas as pd

DAT_PATH = Path(__file__).resolve().parents[0] / "data"

# Imports

import multiprocessing
import concurrent.futures
import functools
import heapq
import re
import typing
from typing import Callable, Literal

import floof
import jarowinkler
import jellyfish
import Levenshtein
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as sklearn_text
import sklearn.neighbors
from thefuzz import fuzz
import tqdm

class Matcher:

    def __init__(self, original: pd.Series, lookup: pd.Series):
        if original.name == lookup.name:
            raise ValueError("Series names must not match.")
        if "score" in (original.name, lookup.name):
            raise ValueError("'score' is a reserved column name.")

        self._original = self._dedupe(original)
        self._lookup = self._dedupe(lookup)

        self._validate()

    def _validate(self):
        if len(self._original.index) == 0 or len(self._lookup.index) == 0:
            raise ValueError("Both series must be non-empty.")

        for col in (self._original, self._lookup):
            if not pd.api.types.is_string_dtype(col):
                raise TypeError(f"Column {col.name} is not string type.")

    @staticmethod
    def _dedupe(series: pd.Series) -> pd.Series:
        series = (
            series.drop_duplicates()
                  .replace({"": pd.NA})
                  .dropna()
        )

        return series

    @staticmethod
    def _clean_and_filter(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
        df["score"] = df["score"] * 100
        df = df.loc[df["score"] >= threshold]

        return df

    @staticmethod
    def _get_score_from_distance(
        lookup_str: str,
        original_str: str,
        original_str_len: int,
        scorer: Callable
    ) -> float:
        dist = scorer(original_str, lookup_str)

        # For methods that return a distance instead of a percentage, divide the returned distance
        # by the length of the longest string to obtain a relative percentage. Otherwise, the
        # distance when comparing long string to short strings will be biased against long srings.
        max_len = max(original_str_len, len(lookup_str))
        pct = (max_len - dist) / max_len

        return (pct, original_str, lookup_str)

    def _get_matches(self, o_str, lookup, scorer, k_matches):
        o_str_len = len(o_str)
        matches = [
            self._get_score_from_distance(lu_str, o_str, o_str_len, scorer) for lu_str in lookup
        ]

        matches = heapq.nlargest(k_matches, matches)

        return matches

    def _get_matches_distance_threads(self, scorer: Callable, k_matches: int, threshold: int) -> pd.DataFrame:
        original = self._original.to_list() # faster to iterate over list
        lookup = self._lookup.to_list() 
        
        og_colname = self._original.name
        lu_colname = self._lookup.name

        merged = {og_colname: [], lu_colname: [], "score": []}
        with tqdm.tqdm(total=len(self._original.index)) as pbar:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                futures = [
                    pool.submit(self._get_matches, o_str, lookup, scorer, k_matches) \
                        for o_str in original
                ]
                for future in concurrent.futures.as_completed(futures):
                    matches = future.result()
                    for score, o_str, lu_str in matches:
                        merged[og_colname].append(o_str)
                        merged[lu_colname].append(lu_str)
                        merged["score"].append(score)

                    pbar.update(1)

        merged = pd.DataFrame(merged)

        return self._clean_and_filter(merged, threshold)

    def _get_matches_distance_process(self, scorer: Callable, k_matches: int, threshold: int) -> pd.DataFrame:
        original = self._original.to_list() # faster to iterate over list
        lookup = self._lookup.to_list() 
        
        og_colname = self._original.name
        lu_colname = self._lookup.name

        merged = {og_colname: [], lu_colname: [], "score": []}
        with tqdm.tqdm(total=len(self._original.index)) as pbar:
            with concurrent.futures.ProcessPoolExecutor() as pool:
                futures = [
                    pool.submit(self._get_matches, o_str, lookup, scorer, k_matches) \
                        for o_str in original
                ]
                for future in concurrent.futures.as_completed(futures):
                    matches = future.result()
                    for score, o_str, lu_str in matches:
                        merged[og_colname].append(o_str)
                        merged[lu_colname].append(lu_str)
                        merged["score"].append(score)

                    pbar.update(1)

        merged = pd.DataFrame(merged)

        return self._clean_and_filter(merged, threshold)

    def _get_matches_distance_seq(self, scorer: Callable, k_matches: int, threshold: int) -> pd.DataFrame:
        original = self._original.to_list() # faster to iterate over list
        lookup = self._lookup.to_list() 
        
        og_colname = self._original.name
        lu_colname = self._lookup.name

        merged = {og_colname: [], lu_colname: [], "score": []}

        for o_str in tqdm.tqdm(original):
            matches = self._get_matches(o_str, lookup, scorer, k_matches)
            for score, o_str, lu_str in matches:
                merged[og_colname].append(o_str)
                merged[lu_colname].append(lu_str)
                merged["score"].append(score)

        merged = pd.DataFrame(merged)

        return self._clean_and_filter(merged, threshold)

    def levenshtein_thread(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_distance_threads(
            scorer=Levenshtein.distance,
            k_matches=k_matches, 
            threshold=threshold
        )

    def levenshtein_process(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_distance_process(
            scorer=Levenshtein.distance,
            k_matches=k_matches, 
            threshold=threshold
        )

    def levenshtein_seq(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_distance_seq(
            scorer=Levenshtein.distance,
            k_matches=k_matches, 
            threshold=threshold
        )

def main():
    df = pd.read_csv(DAT_PATH / "test_data.csv")
    df = df.sample(20_000)

    matcher = Matcher(df["name1"], df["name2"])
    floof_match = floof.Matcher(df["name1"], df["name2"])

    cutils.time_func(matcher.levenshtein_thread)
    cutils.time_func(matcher.levenshtein_process)
    # cutils.time_func(floof_match.levenshtein)
    # cutils.time_func(matcher.levenshtein_seq)

if __name__ == "__main__":
    main()