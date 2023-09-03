import concurrent.futures
import heapq
import typing
from collections.abc import Callable

import jarowinkler
import jellyfish
import Levenshtein
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as sklearn_text
import sklearn.neighbors
import tqdm
from thefuzz import fuzz

from .utils.types import (
    AllScorers,
    EditDistanceScorers,
    NearestNeighborScorers,
    PhoneticScorers,
)
from .utils.utils import _get_ngrams, _get_score, _get_score_from_distance, _normalize


class Matcher:
    def __init__(self, original, lookup, quiet: bool = False):
        self._original = pd.Series(original)
        self._lookup = pd.Series(lookup)
        self._quiet = quiet

        self._set_names()
        self._dedupe()
        self._validate()

        self._original_list = self._original.to_list()
        self._lookup_list = self._lookup.to_list()

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

    def tfidf(self, k_matches: int = None, threshold: int = 0, ngram_length: int = 3):
        original = self._original_list
        lookup = self._lookup_list

        def ngrams_user(string: str, n: int = ngram_length) -> list[str]:
            ngrams = zip(*[string[i:] for i in range(n)])

            return ["".join(ngram) for ngram in ngrams]

        vectorizer = sklearn_text.TfidfVectorizer(min_df=1, analyzer=ngrams_user)
        tf_idf_lookup = vectorizer.fit_transform(lookup)

        if k_matches is None:
            k_matches = min(len(original), len(lookup))

        nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=k_matches, n_jobs=-1, metric="cosine"
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
                "score": confidence_list,
            }
        )

        return self._clean_and_filter(merged, threshold)

    def exact(self) -> pd.DataFrame:
        final = pd.merge(
            left=self._original,
            right=self._lookup,
            how="inner",
            left_on=self._original.name,
            right_on=self._lookup.name,
        )
        final["score"] = 100

        return final

    def _get_matches_phonetic(self, scorer: Callable, name: str) -> pd.DataFrame:
        original_soundex = pd.Series(
            scorer(o_str)
            for o_str in tqdm.tqdm(self._original_list, disable=self._quiet)
        )
        lookup_soundex = pd.Series(
            scorer(lu_str)
            for lu_str in tqdm.tqdm(self._lookup_list, disable=self._quiet)
        )

        original = pd.concat([original_soundex, self._original], axis=1)
        lookup = pd.concat([lookup_soundex, self._lookup], axis=1)

        merged = pd.merge(left=original, right=lookup, how="inner", on=name)
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
        return self._phonetic_match(jellyfish.match_rating_codex, "match_rating_codex")

    def _get_matches_distance(
        self, o_str: str, lookup: list[str], scorer: Callable, k_matches: int
    ) -> list[tuple]:
        matches = [_get_score_from_distance(o_str, lu_str, scorer) for lu_str in lookup]
        matches = heapq.nlargest(k_matches, matches)

        return matches

    def _get_matches_pct(
        self, o_str: str, lookup: list[str], scorer: Callable, k_matches: int
    ) -> list[tuple]:
        matches = [_get_score(o_str, lu_str, scorer) for lu_str in lookup]
        matches = heapq.nlargest(k_matches, matches)

        return matches

    def _get_all_matches(
        self,
        match_func: Callable,
        scorer: Callable,
        k_matches: int,
        threshold: int,
        already_ratio: bool = False,
        ncpus: int = None,
    ) -> pd.DataFrame:
        original = self._original_list  # faster to iterate over list
        lookup = self._lookup_list

        og_colname = self._original.name
        lu_colname = self._lookup.name

        merged = {og_colname: [], lu_colname: [], "score": []}

        with tqdm.tqdm(total=len(self._original.index)) as pbar:
            if ncpus == 1:  # this is just single-threading
                for o_str in original:
                    matches = match_func(o_str, lookup, scorer, k_matches)
                    for score, o_str, lu_str in matches:
                        merged[og_colname].append(o_str)
                        merged[lu_colname].append(lu_str)
                        merged["score"].append(score)

                    pbar.update(1)
            else:
                with concurrent.futures.ProcessPoolExecutor(ncpus) as pool:
                    futures = [
                        pool.submit(match_func, o_str, lookup, scorer, k_matches)
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

        if already_ratio:
            merged["score"] = merged["score"] / 100

        return self._clean_and_filter(merged, threshold)

    def damerau_levenshtein(
        self,
        k_matches: int = 5,
        threshold: int = 80,
        ncpus: int = None,
    ) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_distance,
            scorer=jellyfish.damerau_levenshtein_distance,
            k_matches=k_matches,
            threshold=threshold,
            ncpus=ncpus,
        )

    def levenshtein(
        self, k_matches: int = 5, threshold: int = 80, ncpus: int = None
    ) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_distance,
            scorer=Levenshtein.distance,
            k_matches=k_matches,
            threshold=threshold,
            ncpus=ncpus,
        )

    def hamming(
        self,
        k_matches: int = 5,
        threshold: int = 80,
        ncpus: int = None,
    ) -> pd.DataFrame:
        scorer = jellyfish.hamming_distance

        return self._get_all_matches(
            match_func=self._get_matches_distance,
            scorer=scorer,
            k_matches=k_matches,
            threshold=threshold,
            ncpus=ncpus,
        )

    def jaro_winkler(
        self, k_matches: int = 5, threshold: int = 80, ncpus: int = None
    ) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=jarowinkler.jarowinkler_similarity,
            k_matches=k_matches,
            threshold=threshold,
            ncpus=ncpus,
        )

    def jaro(
        self, k_matches: int = 5, threshold: int = 80, ncpus: int = None
    ) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=jarowinkler.jaro_similarity,
            k_matches=k_matches,
            threshold=threshold,
            ncpus=ncpus,
        )

    def ratio(
        self, k_matches: int = 5, threshold: int = 80, ncpus: int = None
    ) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=fuzz.ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True,
            ncpus=ncpus,
        )

    def partial_ratio(
        self, k_matches: int = 5, threshold: int = 80, ncpus: int = None
    ) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=fuzz.partial_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True,
            ncpus=ncpus,
        )

    def token_sort_ratio(
        self, k_matches: int = 5, threshold: int = 80, ncpus: int = None
    ) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=fuzz.token_sort_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True,
            ncpus=ncpus,
        )

    def token_set_ratio(
        self, k_matches: int = 5, threshold: int = 80, ncpus: int = None
    ) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=fuzz.token_set_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True,
            ncpus=ncpus,
        )

    def partial_token_set_ratio(
        self, k_matches: int = 5, threshold: int = 80, ncpus: int = None
    ) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=fuzz.partial_token_set_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True,
            ncpus=ncpus,
        )

    def partial_token_sort_ratio(
        self, k_matches: int = 5, threshold: int = 80, ncpus: int = None
    ) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=fuzz.partial_token_sort_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True,
            ncpus=ncpus,
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
        scorers: AllScorers = None,
        weights: list[float] = None,
        k_matches: int = 5,
        threshold: int = 80,
        filter: bool = True,
        filter_k_matches: int = 20,
        filter_threshold: int = 50,
        drop_intermediate: bool = True,
        ncpus: int = None,
    ) -> pd.DataFrame:
        """A convenience function to iterate through multiple different matching
        algorithms and generate a combined score / crosswalk.

        Parameters
        ----------
        scorers : AllScorers, optional
            A list of scorers to use. The available scorers are:
                - damerau_levenshtein
                - hamming
                - jaro
                - jaro_winkler
                - levenshtein
                - partial_ratio
                - partial_token_set_ratio
                - partial_token_sort_ratio
                - ratio
                - token_set_ratio
                - token_sort_ratio
                - tfidf
                - match_rating_codex
                - metaphone
                - nysiis
                - soundex,
            by default None
        weights : list[float], optional
            How much to weigh each algorithm, by default None
        k_matches : int, optional
            Up to how many matches should be returned, by default 5
        threshold : int, optional
            Keep only matches above this score. Note that this applies only to the FINAL
            calculated score. This is because if it applied at each individual step,
            information would be lost. For example, a Hamming score of 79 provides more
            information than a Hamming score of 60, but both would be set to 0 if it
            applied to each individual step, by default 80
        filter : bool, optional
            If True, uses TFIDF to filter out unlikely matches, by default True
        filter_k_matches : int, optional
            _description_
        filter_threshold : int, optional
            Threshold to use for the TFIDF filter, by default 20, by default 50
        drop_intermediate : bool, optional
            Drop intermediate columns, keeping only the final calculated score,
            by default True
        ncpus : int, optional
            _description_, by default None

        Returns
        -------
        pd.DataFrame
            A crosswalk of the two columns + the score

        Raises
        ------
        ValueError
            If the number of scorers and weights don't match
        ValueError
            If there are no valid lookups
        ValueError
            If the scorer name is not in the accepted list
        """
        # Variables
        og_colname = self._original.name
        lu_colname = self._lookup.name

        # Define scorers and weights
        if weights is None and scorers is None:
            default_scorers = {
                "damerau_levenshtein": 1,
                "jaro": 0.9,
                "jaro_winkler": 1,
                "levenshtein": 0.9,
                "tfidf": 0.8,
            }
            scorers = default_scorers.keys()
            weights = default_scorers.values()
        elif weights is None and scorers is not None:
            weights = [1 for _ in scorers]

        if len(weights) != len(scorers):
            raise ValueError("Number of scorers and weights must match.")

        weights = _normalize(weights)
        scorers_dict = {s: w for s, w in zip(scorers, weights)}

        # Start with an exact match
        final = self.exact()
        final = final.rename(columns={"score": "exact_score"})

        # Remove exact matches from the possible pool
        mask = ~(self._lookup.isin(set(final[lu_colname])))
        self._lookup = self._lookup.loc[mask]

        # Filter using tfidf
        if filter:
            # Use tfidf match, getting max number of neighbors, and applying a generous
            # default threshold of 50
            try:
                tfidf = self.tfidf(
                    k_matches=filter_k_matches, threshold=filter_threshold
                )
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
                df = func(k_matches=k_matches, threshold=0, ncpus=ncpus)
            elif scorer_nm in nn_scorers:
                # If we have already filtered using tfidf, no sense in doing it again
                if scorer_nm == "tfidf" and filter and filter_threshold >= threshold:
                    df = tfidf
                else:
                    # With nearest neighbor, k_matches is not UP to k_matches, it is
                    # that number of neighbors exactly, which can cause the algorithm to
                    # error. If this happens, it is likely that user wants up to
                    # k_matches, so try again with the "None" option, which sets
                    # k_matches to the maximum possible value.
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
                    "score_adj": f"{scorer_nm}_score_adj",
                }
            )
            score_cols.append(f"{scorer_nm}_score_adj")

            final = pd.merge(
                left=final,
                right=df,
                how="outer",  # keep both since diff algs may return diff matches
                on=[og_colname, lu_colname],
            )

        # Finally, calculate the score
        final["final_score"] = final[score_cols].sum(axis=1, skipna=True)
        final["final_score"] = np.where(
            final["exact_score"] == 100, 100, final["final_score"]
        )
        final = final.loc[final["final_score"] >= threshold]

        if drop_intermediate:
            final = final[[og_colname, lu_colname, "final_score"]]

        return final
