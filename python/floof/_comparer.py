from collections.abc import Callable
from typing import Union

import jarowinkler
import jellyfish
import Levenshtein
import pandas as pd
import tqdm
from thefuzz import fuzz

from .utils.types import EditDistanceScorers, PhoneticScorers
from .utils.utils import (
    _get_score,
    _get_score_from_distance,
    _get_score_phonetic,
    _normalize,
)


class Comparer:
    def __init__(
        self,
        original,
        lookup,
        quiet: bool = False,
    ):
        self._original = pd.Series(original)
        self._lookup = pd.Series(lookup)
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

    def damerau_levenshtein(self) -> list[float]:
        return self._apply_score(
            match_func=_get_score_from_distance,
            scorer=jellyfish.damerau_levenshtein_distance,
        )

    def levenshtein(self) -> list[float]:
        return self._apply_score(
            match_func=_get_score_from_distance, scorer=Levenshtein.distance
        )

    def hamming(self) -> list[float]:
        return self._apply_score(
            match_func=_get_score_from_distance, scorer=jellyfish.hamming_distance
        )

    def jaro_winkler(self) -> list[float]:
        return self._apply_score(
            match_func=_get_score, scorer=jarowinkler.jarowinkler_similarity
        )

    def jaro(self) -> list[float]:
        return self._apply_score(
            match_func=_get_score, scorer=jarowinkler.jaro_similarity
        )

    def ratio(self) -> list[float]:
        return self._apply_score(
            match_func=_get_score, scorer=fuzz.ratio, already_ratio=True
        )

    def partial_ratio(self) -> list[float]:
        return self._apply_score(
            match_func=_get_score, scorer=fuzz.partial_ratio, already_ratio=True
        )

    def token_sort_ratio(self) -> list[float]:
        return self._apply_score(
            match_func=_get_score, scorer=fuzz.token_sort_ratio, already_ratio=True
        )

    def token_set_ratio(self) -> list[float]:
        return self._apply_score(
            match_func=_get_score, scorer=fuzz.token_set_ratio, already_ratio=True
        )

    def partial_token_set_ratio(self) -> list[float]:
        return self._apply_score(
            match_func=_get_score,
            scorer=fuzz.partial_token_set_ratio,
            already_ratio=True,
        )

    def partial_token_sort_ratio(self) -> list[float]:
        return self._apply_score(
            match_func=_get_score,
            scorer=fuzz.partial_token_sort_ratio,
            already_ratio=True,
        )

    def soundex(self) -> list[int]:
        return self._apply_score(
            match_func=_get_score_phonetic, scorer=jellyfish.soundex, already_ratio=True
        )

    def nysiis(self) -> list[int]:
        return self._apply_score(
            match_func=_get_score_phonetic, scorer=jellyfish.nysiis, already_ratio=True
        )

    def match_rating_codex(self) -> list[int]:
        return self._apply_score(
            match_func=_get_score_phonetic,
            scorer=jellyfish.match_rating_codex,
            already_ratio=True,
        )

    def metaphone(self) -> list[int]:
        return self._apply_score(
            match_func=_get_score_phonetic,
            scorer=jellyfish.metaphone,
            already_ratio=True,
        )

    def _dispatcher(self, func_name: str) -> Callable:
        func_mapper = {
            "damerau_levenshtein": self.damerau_levenshtein,
            "hamming": self.hamming,
            "jaro": self.jaro,
            "jaro_winkler": self.jaro_winkler,
            "levenshtein": self.levenshtein,
            "partial_ratio": self.partial_ratio,
            "partial_token_set_ratio": self.partial_token_set_ratio,
            "partial_token_sort_ratio": self.partial_token_sort_ratio,
            "ratio": self.ratio,
            "token_set_ratio": self.token_set_ratio,
            "token_sort_ratio": self.token_sort_ratio,
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
                "jaro": 0.9,
                "jaro_winkler": 1,
                "levenshtein": 0.9,
                "partial_ratio": 0.5,
                "token_sort_ratio": 0.6,
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
