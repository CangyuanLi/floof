# Imports

import concurrent.futures
import functools
import heapq
import re
import typing
from typing import Callable, Literal

import jarowinkler
import jellyfish
import Levenshtein
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as sklearn_text
import sklearn.neighbors
from thefuzz import fuzz

EditDistanceScorers = Literal[
    "damerau_levenshtein",
    "hamming",
    "jaro",
    "jaro_winkler",
    "levenshtein",
    "partial_ratio",
    "partial_token_set_ratio",
    "partial_token_sort_ratio",
    "ratio",
    "token_set_ratio",
    "token_sort_ratio",
]

NearestNeighborScorers = Literal[
    "tfidf",
]

PhoneticScorers = Literal[
    "match_rating_codex",
    "metaphone",
    "nysiis",
    "soundex",
]

AllScorers = Literal[EditDistanceScorers, NearestNeighborScorers, PhoneticScorers]

DEFAULT_SCORERS = {
    "damerau_levenshtein": 1, 
    "jaro": .9, 
    "jaro_winkler": 1, 
    "levenshtein": .9,
    "tfidf": .8
}

class Matcher:

    def __init__(self, original: pd.Series, lookup: pd.Series) -> None:
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
    def get_ngrams(string: str, n: int=3, strip_punc: bool=True) -> list[str]:
        if strip_punc:
            pattern = re.compile(r"[,-./]|\sBD")
            string = re.sub(pattern, "", string)

        ngrams = zip(*[string[i:] for i in range(n)])

        return ["".join(ngram) for ngram in ngrams]

    def tfidf(self, k_matches: int=None, threshold: int=0, ngram_length: int=3):
        original = self._original.to_list()
        lookup = self._lookup.to_list()

        def ngrams_user(string: str, n: int=ngram_length) -> str:
            return self.get_ngrams(string, n)

        vectorizer = sklearn_text.TfidfVectorizer(min_df=1, analyzer=ngrams_user)
        tf_idf_lookup = vectorizer.fit_transform(lookup)

        if k_matches is None:
            k_matches = min(len(original), len(lookup))

        nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=k_matches,
            n_jobs=-1,
            metric="cosine"
        ).fit(tf_idf_lookup)

        tf_idf_original = vectorizer.transform(original)
        distances, lookup_indices = nbrs.kneighbors(tf_idf_original)

        original_name_list = []
        confidence_list = []
        lookup_list = []

        for i, lookup_index in enumerate(lookup_indices):
            original_name = original[i]
            for j in lookup_index:
                original_name_list.append(original_name)
                lookup_list.append(lookup[j])

            for dist in distances[i]:
                confidence_list.append(1 - dist)

        merged = pd.DataFrame(
            {
                self._original.name: original_name_list,
                self._lookup.name: lookup_list,
                "score": confidence_list
            }
        )

        return self._clean_and_filter(merged, threshold)

    def exact(self) -> pd.DataFrame:
        final = pd.merge(
            left=self._original,
            right=self._lookup,
            how="inner",
            left_on=self._original.name,
            right_on=self._lookup.name
        )
        final["score"] = 100

        return final

    def _phonetic_match(self, scorer: Callable, name: str) -> pd.DataFrame:
        original_soundex = self._original.apply(scorer).rename(name)
        lookup_soundex = self._lookup.apply(scorer).rename(name)

        original = pd.concat([original_soundex, self._original], axis=1)
        lookup = pd.concat([lookup_soundex, self._lookup], axis=1)

        merged = pd.merge(
            left=original,
            right=lookup,
            how="inner",
            on=name
        )
        merged["score"] = 100
        merged.drop(columns=[name], inplace=True)

        return merged

    def soundex(self) -> pd.DataFrame:
        return self._phonetic_match(jellyfish.soundex, "soundex")

    def metaphone(self) -> pd.DataFrame:
        return self._phonetic_match(jellyfish.metaphone, "metaphone")

    def nysiis(self) -> pd.DataFrame:
        return self._phonetic_match(jellyfish.nysiis, "nysiis")

    def match_rating_codex(self) -> pd.DataFrame:
        return self._phonetic_match(jellyfish.match_rating_codex, "mr_codex")

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

        return (pct, lookup_str)

    @staticmethod
    def _get_score(lookup_str: str, original_str: str, scorer: Callable) -> float:
        pct = scorer(original_str, lookup_str)

        return (pct, lookup_str)

    def _get_matches_pct(
        self, 
        scorer: Callable, 
        k_matches: int, 
        threshold: int,
        already_ratio: bool=False
    ) -> pd.DataFrame:
        original = self._original
        lookup = self._lookup
        
        # Maybe premature optimization, but let's store all the things we will need in variables
        # up front, instead of paying the cost of, e.g. lookup in the loop
        og_colname = original.name
        lu_colname = lookup.name

        merged = {og_colname: [], lu_colname: [], "score": []}
        for o_str in original:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                func = functools.partial(
                    self._get_score,
                    original_str=o_str,
                    scorer=scorer
                )
                matches = pool.map(func, lookup)

            matches = heapq.nlargest(k_matches, matches)
            for score, lu_name in matches:
                merged[og_colname].append(o_str)
                merged[lu_colname].append(lu_name)
                merged["score"].append(score)

        merged = pd.DataFrame(merged)

        if already_ratio:
            merged["score"] = merged["score"] / 100

        return self._clean_and_filter(merged, threshold)

    def _get_matches_distance(self, scorer: Callable, k_matches: int, threshold: int) -> pd.DataFrame:
        original = self._original
        lookup = self._lookup
        
        # Maybe premature optimization, but let's store all the things we will need in variables
        # up front, instead of paying the cost of, e.g. lookup in the loop
        og_colname = original.name
        lu_colname = lookup.name

        merged = {og_colname: [], lu_colname: [], "score": []}
        for o_str in original:
            o_str_len = len(o_str)
            with concurrent.futures.ThreadPoolExecutor() as pool:
                func = functools.partial(
                    self._get_score_from_distance,
                    original_str=o_str,
                    original_str_len=o_str_len,
                    scorer=scorer
                )
                matches = pool.map(func, lookup)

            matches = heapq.nlargest(k_matches, matches)
            for score, lu_name in matches:
                merged[og_colname].append(o_str)
                merged[lu_colname].append(lu_name)
                merged["score"].append(score)

        merged = pd.DataFrame(merged)

        return self._clean_and_filter(merged, threshold)

    def damerau_levenshtein(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_distance(
            scorer=jellyfish.damerau_levenshtein_distance,
            k_matches=k_matches, 
            threshold=threshold
        )

    def levenshtein(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_distance(
            scorer=Levenshtein.distance,
            k_matches=k_matches, 
            threshold=threshold
        )

    def hamming(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_pct(
            scorer=jellyfish.hamming_distance,
            k_matches=k_matches,
            threshold=threshold
        )

    def jaro_winkler(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_pct(
            scorer=jarowinkler.jarowinkler_similarity,
            k_matches=k_matches,
            threshold=threshold
        )

    def jaro(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_pct(
            scorer=jarowinkler.jaro_similarity,
            k_matches=k_matches,
            threshold=threshold
        )

    def ratio(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_pct(
            scorer=fuzz.ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True
        )

    def partial_ratio(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_pct(
            scorer=fuzz.partial_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True
        )

    def token_sort_ratio(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_pct(
            scorer=fuzz.token_sort_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True
        )

    def token_set_ratio(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_pct(
            scorer=fuzz.token_set_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True
        )

    def partial_token_set_ratio(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_pct(
            scorer=fuzz.partial_token_set_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True
        )

    def partial_token_sort_ratio(self, k_matches: int=5, threshold: int=80) -> pd.DataFrame:
        return self._get_matches_pct(
            scorer=fuzz.partial_token_sort_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True
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
            "tfidf": self.tfidf,
            "match_rating_codex": self.match_rating_codex,
            "metaphone": self.metaphone,
            "nysiis": self.nysiis,
            "soundex": self.soundex,
        }

        return func_mapper[func_name]

    def match(
        self,
        scorers: AllScorers=None,
        weights: list[int | float]=None,
        k_matches: int=5,
        threshold: int=80,
        filter: bool=True,
        filter_k_matches: int=20,
        filter_threshold: int=50,
        drop_intermediate: bool=True
    ) -> pd.DataFrame:
        """
        A convenience function to iterate through multiple different matching algorithms and
        generate a combined score / crosswalk. It always begins with an exact match.

        Args:
            scorers (AllScorers, optional): Defaults to None.
                A list of scorers to use. 
                The available scorers are:
                    'damerau_levenshtein'
                    'hamming'
                    'jaro'
                    'jaro_winkler'
                    'levenshtein'
                    'partial_ratio'
                    'partial_token_set_ratio'
                    'partial_token_sort_ratio'
                    'ratio'
                    'token_set_ratio'
                    'token_sort_ratio'
                    'tfidf'
                    'match_rating_codex'
                    'metaphone'
                    'nysiis'
                    'soundex'
            weights (list[int  |  float], optional): Defaults to None.
                How much to weigh each algorithm. 
            k_matches (int, optional): Defaults to 5.
                Up to how many matches should be returned. 
            threshold (int, optional): Defaults to 80.
                Keep only matches above this score. Note that this applies only to the FINAL
                calculated score. This is because if it applied at each individual step, information
                would be lost. For example, a Hamming score of 79 provides more information than
                a Hamming score of 60, but both would be set to 0 if it applied to each individual
                step. 
            filter (bool, optional): Defaults to True.
                If True, uses TFIDF to filter out unlikely matches. 
            filter_threshold (int, optional): Defaults to 50.
                Threshold to use for the TFIDF filter. 
            drop_intermediate (bool, optional): Defaults to True. 
                Drop intermediate columns, keeping only the final calculated score. 
                
        Raises:
            ValueError: If the number of scorers and weights don't match.
            ValueError: If there are no valid lookups.
            ValueError: If the scorer name is not in the accepted list.

        Returns:
            pd.DataFrame: A crosswalk of the two columns + the score.

        Example:
            Name1   Name2   Jaro    Hamming
            Apple   Appl    90      80
            Apple   Apl     80      70
            Bin     Tot     0       0
        """
        # Variables
        og_colname = self._original.name
        lu_colname = self._lookup.name

        # Define scorers and weights
        if weights is None and scorers is None:
            scorers = DEFAULT_SCORERS.keys()
            weights = DEFAULT_SCORERS.values()
        elif weights is None and scorers is not None:
            weights = [1 for _ in scorers]

        if len(weights) != len(scorers):
            raise ValueError("Number of scorers and weights must match.")

        sum_w = sum(weights)
        weights = [w / sum_w for w in weights] # normalize to 1
        scorers_dict = {s: w for s, w in zip(scorers, weights)}

        # Start with an exact match
        final = self.exact()
        final = final.rename(columns={"score": "exact_score"})

        # Remove exact matches from the possible pool
        mask = ~(self._lookup.isin(set(final[lu_colname])))
        self._lookup = self._lookup.loc[mask]

        # Filter using tfidf
        if filter:
            # Use tfidf match, getting max number of neighbors, and applying a generous default
            # threshold of 50
            try:
                tfidf = self.tfidf(k_matches=filter_k_matches, threshold=filter_threshold)
            except ValueError:
                tfidf = self.tfidf(k_matches=None, threshold=filter_threshold)

            self._lookup = self._lookup.loc[self._lookup.isin(set(tfidf[lu_colname]))]

        # Check if we have the required number observations on both sides
        num_exact_matches = len(final.index)
        lookups_left = len(self._lookup.index)
        if num_exact_matches == 0 and lookups_left == 0:
            raise ValueError("No valid lookups.")
        elif num_exact_matches > 0 and lookups_left == 0:
            return final

        # Match
        edit_scorers = typing.get_args(EditDistanceScorers)
        nn_scorers = typing.get_args(NearestNeighborScorers)
        phonetic_scorers = typing.get_args(PhoneticScorers)
        score_cols = []
        for scorer_nm, weight in scorers_dict.items():
            func = self._dispatcher(scorer_nm)

            if scorer_nm in edit_scorers:
                df = func(k_matches=k_matches, threshold=0)
            elif scorer_nm in nn_scorers:
                # If we have already filtered using tfidf, no sense in doing it again
                if scorer_nm == "tfidf" and filter and filter_threshold >= threshold:
                    df = tfidf
                else:
                    # With nearest neighbor, k_matches is not UP to k_matches, it is that number of
                    # neighbors exactly, which can cause the algorithm to error. If this happens, it
                    # is likely that user wants up to k_matches, so try again with the "None" 
                    # option, which sets k_matches to the maximum possible value.
                    try:
                        df = func(k_matches=k_matches, threshold=0)
                    except ValueError:
                        df = func(k_matches=None, threshold=0)
            elif scorer_nm in phonetic_scorers:
                df = func()
            else:
                raise ValueError("Scorer name is not valid.")

            df["score_adj"] = df["score"] * weight
            df = df.rename(
                columns={
                    "score": f"{scorer_nm}_score",
                    "score_adj": f"{scorer_nm}_score_adj"
                }
            )
            score_cols.append(f"{scorer_nm}_score_adj")

            final = pd.merge(
                left=final,
                right=df,
                how="outer", # keep both since different algs may return different matches
                on=[og_colname, lu_colname]
            )

        # Finally, calculate the score
        final["final_score"] = final[score_cols].sum(axis=1, skipna=True)
        final["final_score"] = np.where(final["exact_score"] == 100, 100, final["final_score"])
        final = final.loc[final["final_score"] >= threshold]

        if drop_intermediate:
            final = final[[og_colname, lu_colname, "final_score"]]

        return final
