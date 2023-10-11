from collections.abc import Callable

import numpy as np
import pandas as pd
from thefuzz import fuzz

from ._rustyfloof import (
    _extract_bytes_tup,
    _extract_graphemes_tup,
    _match,
    _match_slice,
    _match_slice_ascii,
)


class Matcher:
    def __init__(
        self,
        original,
        lookup,
        ascii_only: bool = False,
        n_jobs: int = -1,
        quiet: bool = False,
    ):
        # Set class variables
        self._original = pd.Series(original)
        self._lookup = pd.Series(lookup)
        self._n_jobs = 0 if n_jobs == -1 else n_jobs
        self._ascii_only = ascii_only
        self._quiet = quiet

        self._set_names()
        self._dedupe()
        self._validate()

        # It's faster to iterate over lists on the python end, and provides a nice way
        # to interface with the Rust backend as well.
        self._original_list = self._original.to_list()
        self._lookup_list = self._lookup.to_list()

        # The custom Rust implementations have functions that operate on an arbitrary
        # slice. This avoids the work of doing e.g. unicode grapheme segmentation on
        # every function call, which is what happens if you just call func(s1, s2) in
        # a nested loop.
        func = _extract_bytes_tup if self._ascii_only else _extract_graphemes_tup
        self._match_slice_func = (
            _match_slice_ascii if self._ascii_only else _match_slice
        )

        self._original_list_processed = func(self._original_list)
        self._lookup_list_processed = func(self._lookup_list)

    def _set_names(self):
        if self._original.name is None:
            self._original.name = "original"

        if self._lookup.name is None:
            self._lookup.name = "lookup"

        if self._original.name == self._lookup.name:
            raise ValueError("Series names must not match.")
        if "score" in (self._original.name, self._lookup.name):
            raise ValueError("'score' is a reserved column name.")

    def _validate(self):
        if len(self._original.index) == 0 or len(self._lookup.index) == 0:
            raise ValueError("Both series must be non-empty.")

        for col in (self._original, self._lookup):
            if not pd.api.types.is_string_dtype(col):
                raise TypeError(f"Column {col.name} is not string type.")

    def _dedupe(self):
        # there may be a mix of NaN and empty strings, so the easiest way is to just
        # replace one with the other (NaN with "" or vice versa) first
        self._original = self._original.drop_duplicates().replace({"": pd.NA}).dropna()
        self._lookup = self._lookup.drop_duplicates().replace({"": pd.NA}).dropna()

    @staticmethod
    def _clean_and_filter(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
        df["score"] = df["score"] * 100
        df = df.loc[df["score"] >= threshold]

        return df

    def exact(self) -> pd.DataFrame:
        final = pd.merge(
            left=self._original,
            right=self._lookup,
            how="inner",
            left_on=self._original.name,
            right_on=self._lookup.name,
        )
        final["score"] = 1

        return final

    def soundex(self, k_matches: int = 5, threshold: float = 0) -> pd.DataFrame:
        scorer = "soundex_ascii" if self._ascii_only else "soundex"

        return self._get_all_matches_rust(scorer, k_matches, threshold)

    def _get_all_matches_rust(
        self, scorer: str, k_matches: int, threshold: float
    ) -> pd.DataFrame:
        return pd.DataFrame(
            _match(
                self._original_list,
                self._lookup_list,
                scorer,
                k_matches,
                threshold,
                self._n_jobs,
                self._quiet,
            ),
            columns=["score", self._original.name, self._lookup.name],
        )

    def _get_all_matches_rust_slice(
        self, scorer: str, k_matches: int, threshold: float
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self._match_slice_func(
                self._original_list_processed,
                self._lookup_list_processed,
                scorer,
                k_matches,
                threshold,
                self._n_jobs,
                self._quiet,
            )
        )

    def damerau_levenshtein(
        self,
        k_matches: int = 5,
        threshold: float = 0,
    ) -> pd.DataFrame:
        scorer = (
            "damerau_levenshtein_similarity_ascii"
            if self._ascii_only
            else "damerau_levenshtein_similarity"
        )

        return self._get_all_matches_rust_slice(scorer, k_matches, threshold)

    def levenshtein(self, k_matches: int = 5, threshold: float = 0) -> pd.DataFrame:
        return self._get_all_matches_rust_slice(
            "levenshtein_similarity", k_matches, threshold
        )

    def osa(self, k_matches: int = 5, threshold: int = 0) -> pd.DataFrame:
        return self._get_all_matches_rust_slice("osa_similarity", k_matches, threshold)

    def hamming(
        self,
        k_matches: int = 5,
        threshold: float = 0,
    ) -> pd.DataFrame:
        # Usually, pre-processing the strings results in a significant speed increase,
        # even for only ascii strings. However, this is not the case for Hamming, maybe
        # due to the simplicity of the algorithm?
        if self._ascii_only:
            return self._get_all_matches_rust("hamming_ascii", k_matches, threshold)

        return self._get_all_matches_rust_slice(
            "hamming_similarity", k_matches, threshold
        )

    def jaccard(self, k_matches: int = 5, threshold: float = 0) -> pd.DataFrame:
        return self._get_all_matches_rust_slice(
            "jaccard_similarity", k_matches, threshold
        )

    def sorensen_dice(self, k_matches: int = 5, threshold: float = 0) -> pd.DataFrame:
        return self._get_all_matches_rust_slice(
            "sorensen_dice_similarity", k_matches, threshold
        )

    def cosine(self, k_matches: int = 5, threshold: float = 0) -> pd.DataFrame:
        return self._get_all_matches_rust_slice(
            "cosine_similarity", k_matches, threshold
        )

    def bag(self, k_matches: int = 5, threshold: float = 0) -> pd.DataFrame:
        return self._get_all_matches_rust_slice("bag_similarity", k_matches, threshold)

    def overlap(self, k_matches: int = 5, threshold: float = 0) -> pd.DataFrame:
        return self._get_all_matches_rust_slice(
            "overlap_similarity", k_matches, threshold
        )

    def tversky(self, k_matches: int = 5, threshold: float = 0) -> pd.DataFrame:
        return self._get_all_matches_rust_slice(
            "tversky_similarity", k_matches, threshold
        )

    def jaro_winkler(self, k_matches: int = 5, threshold: float = 0) -> pd.DataFrame:
        return self._get_all_matches_rust_slice(
            "jaro_winkler_similarity", k_matches, threshold
        )

    def jaro(
        self,
        k_matches: int = 5,
        threshold: float = 0,
    ) -> pd.DataFrame:
        return self._get_all_matches_rust_slice("jaro_similarity", k_matches, threshold)

    def _dispatcher(self, func_name: str) -> Callable:
        func_mapper = {
            "damerau_levenshtein": self.damerau_levenshtein,
            "hamming": self.hamming,
            "jaro": self.jaro,
            "jaro_winkler": self.jaro_winkler,
            "levenshtein": self.levenshtein,
            "soundex": self.soundex,
        }

        return func_mapper[func_name]
