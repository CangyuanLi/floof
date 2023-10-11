from collections.abc import Callable
from typing import Union

import pandas as pd
import tqdm

from ._rustyfloof import _compare
from .utils.types import EditDistanceScorers, PhoneticScorers
from .utils.utils import _normalize


class Comparer:
    def __init__(
        self,
        original,
        lookup,
        ascii_only: bool = False,
        n_jobs: int = 1,
        quiet: bool = False,
    ):
        self._original = pd.Series(original)
        self._lookup = pd.Series(lookup)
        self._ascii_only = ascii_only
        self._n_jobs = 0 if n_jobs == -1 else n_jobs
        self._quiet = quiet

        self._validate()
        self._normalize()

        self._original = self._original.to_list()
        self._lookup = self._lookup.to_list()

    def _validate(self):
        og_len = len(self._original.index)
        lu_len = len(self._lookup.index)

        if og_len == 0 or lu_len == 0:
            raise ValueError("Both series must be non-empty.")

        if og_len != lu_len:
            raise ValueError("Column lengths do not match.")

        for col in (self._original, self._lookup):
            if not pd.api.types.is_string_dtype(col):
                raise TypeError(f"Column {col.name} is not string type.")

    def _normalize(self):
        self._original = self._original.fillna("")
        self._lookup = self._lookup.fillna("")

    def _apply_score(
        self, match_func: Callable, scorer: Callable, already_ratio: bool = False
    ) -> list[float]:
        score_lst = []
        with tqdm.tqdm(total=len(self._original), disable=self._quiet) as pbar:
            for o_str, lu_str in zip(self._original, self._lookup):
                score, _, _ = match_func(o_str, lu_str, scorer)
                score_lst.append(score)

                pbar.update(1)

        if already_ratio:
            return score_lst

        return [i * 100 for i in score_lst]

    def jaccard(self) -> list[float]:
        scorer = "jaccard_ascii" if self._ascii_only else "jaccard"

        return _compare(self._original, self._lookup, scorer, self._n_jobs)

    def sorensen_dice(self) -> list[float]:
        scorer = "sorensen_dice_ascii" if self._ascii_only else "sorensen_dice"

        return _compare(self._original, self._lookup, scorer, self._n_jobs)

    def cosine(self) -> list[float]:
        scorer = "cosine_ascii" if self._ascii_only else "cosine"

        return _compare(self._original, self._lookup, scorer, self._n_jobs)

    def bag(self) -> list[float]:
        scorer = "bag_ascii" if self._ascii_only else "bag"

        return _compare(self._original, self._lookup, scorer, self._n_jobs)

    def overlap(self) -> list[float]:
        scorer = "overlap_ascii" if self._ascii_only else "overlap"

        return _compare(self._original, self._lookup, scorer, self._n_jobs)

    def tversky(self) -> list[float]:
        scorer = "tversky_ascii" if self._ascii_only else "tversky"

        return _compare(self._original, self._lookup, scorer, self._n_jobs)

    def damerau_levenshtein(self) -> list[float]:
        scorer = (
            "damerau_levenshtein_ascii" if self._ascii_only else "damerau_levenshtein"
        )

        return _compare(self._original, self._lookup, scorer, self._n_jobs)

    def levenshtein(self) -> list[float]:
        scorer = "levenshtein_ascii" if self._ascii_only else "levenshtein"

        return _compare(self._original, self._lookup, scorer, self._n_jobs)

    def osa(self) -> list[float]:
        scorer = "osa_ascii" if self._ascii_only else "osa"

        return _compare(self._original, self._lookup, scorer, self._n_jobs)

    def hamming(self) -> list[float]:
        scorer = "hamming_ascii" if self._ascii_only else "hamming"

        return _compare(self._original, self._lookup, scorer, self._n_jobs)

    def jaro_winkler(self) -> list[float]:
        scorer = "jaro_winkler_ascii" if self._ascii_only else "jaro_winkler"

        return _compare(self._original, self._lookup, scorer, self._n_jobs)

    def jaro(self) -> list[float]:
        scorer = "jaro_ascii" if self._ascii_only else "jaro"

        return _compare(self._original, self._lookup, scorer, self._n_jobs)

    def soundex(self) -> list[float]:
        scorer = "soundex_ascii" if self._ascii_only else "soundex"

        return _compare(self._original, self._lookup, scorer, self._n_jobs)

    def _dispatcher(self, func_name: str) -> Callable:
        func_mapper = {
            "damerau_levenshtein": self.damerau_levenshtein,
            "hamming": self.hamming,
            "jaro": self.jaro,
            "jaro_winkler": self.jaro_winkler,
            "levenshtein": self.levenshtein,
        }

        return func_mapper[func_name]

    def compare(
        self,
        scorers: Union[EditDistanceScorers, PhoneticScorers] = None,
        weights: list[float] = None,
        drop_intermediate: bool = True,
    ) -> list[float] | pd.DataFrame:
        # Define scorers and weights
        if weights is None and scorers is None:
            default_scorers = {
                "damerau_levenshtein": 1,
                "jaro_winkler": 1,
                "hamming": 0.1,
            }
            scorers = default_scorers.keys()
            weights = default_scorers.values()
        elif weights is None and scorers is not None:
            weights = [1 for _ in scorers]

        if len(weights) != len(scorers):
            raise ValueError("Number of scorers and weights must match.")

        weights = _normalize(weights)  # normalize to 1
        scorer_dict = {s: w for s, w in zip(scorers, weights)}

        scores = dict()
        for func_nm, weight in scorer_dict.items():
            func = self._dispatcher(func_nm)
            scores[f"{func_nm}_score"] = [i * weight for i in func()]

        df = pd.DataFrame(scores)
        df["final_score"] = df[list(scores.keys())].sum(axis=1, skipna=True)

        if drop_intermediate:
            return df["final_score"].to_list()

        return df
