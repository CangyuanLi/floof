# Imports

import concurrent.futures
import heapq
import re
import typing
from typing import Callable, Literal

import fast_distance
import jarowinkler
import jellyfish
import Levenshtein
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as sklearn_text
import sklearn.neighbors
from thefuzz import fuzz
import tqdm

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

# Functions used in both classes
def _get_score_from_distance(
    lookup_str: str,
    original_str: str,
    original_str_len: int,
    scorer: Callable
) -> tuple[float, str, str]:
    dist = scorer(original_str, lookup_str)

    # For methods that return a distance instead of a percentage, divide the returned distance
    # by the length of the longest string to obtain a relative percentage. Otherwise, the
    # distance when comparing long string to short strings will be biased against long srings.
    max_len = max(original_str_len, len(lookup_str))
    pct = (max_len - dist) / max_len

    return (pct, original_str, lookup_str)

def _get_score(lookup_str: str, original_str: str, scorer: Callable) -> tuple[float, str, str]:
    pct = scorer(original_str, lookup_str)

    return (pct, original_str, lookup_str)

class Comparer:

    def __init__(self, original: pd.Series, lookup: pd.Series):
        self._original = original
        self._lookup = lookup

        self._validate()

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

    def _get_score_from_distance(self, o_str: str, lu_str: str, scorer: Callable) -> float:
        if o_str == "" or lu_str == "":
            return (0, o_str, lu_str)

        o_str_len = len(o_str)

        return _get_score_from_distance(o_str, lu_str, o_str_len, scorer)

    def _get_score(self, o_str: str, lu_str: str, scorer: Callable) -> float:
        return _get_score(o_str, lu_str, scorer)

    def _apply_score(
        self,
        match_func: Callable,
        scorer: Callable,
        already_ratio: bool=False
    ) -> pd.Series:
        original = self._original.to_list() # faster to iterate over list
        lookup = self._lookup.to_list() 
        
        score_lst = []
        with tqdm.tqdm(total=len(self._original.index)) as pbar:
            for o_str, lu_str in zip(original, lookup):
                score, _, _ = match_func(o_str, lu_str, scorer)
                score_lst.append(score)

                pbar.update(1)

        res = pd.Series(score_lst)
        
        if already_ratio:
            return res

        res = res * 100

        return res

    def damerau_levenshtein(self) -> pd.Series:
        return self._apply_score(
            match_func=self._get_score_from_distance,
            scorer=jellyfish.damerau_levenshtein_distance
        )

    def levenshtein(self) -> pd.Series:
        return self._apply_score(
            match_func=self._get_score_from_distance,
            scorer=Levenshtein.distance
        )

    def hamming(self) -> pd.Series:
        return self._apply_score(
            match_func=self._get_score_from_distance,
            scorer=jellyfish.hamming_distance
        )

    def jaro_winkler(self) -> pd.Series:
        return self._apply_score(
            match_func=self._get_score,
            scorer=jarowinkler.jarowinkler_similarity
        )

    def jaro(self) -> pd.Series:
        return self._apply_score(
            match_func=self._get_score,
            scorer=jarowinkler.jaro_similarity
        )

    def ratio(self) -> pd.Series:
        return self._apply_score(
            match_func=self._get_score,
            scorer=fuzz.ratio,
            already_ratio=True
        )

    def partial_ratio(self) -> pd.Series:
        return self._apply_score(
            match_func=self._get_score,
            scorer=fuzz.partial_ratio,
            already_ratio=True
        )

    def token_sort_ratio(self) -> pd.Series:
        return self._apply_score(
            match_func=self._get_score,
            scorer=fuzz.token_sort_ratio,
            already_ratio=True
        )

    def token_set_ratio(self) -> pd.Series:
        return self._apply_score(
            match_func=self._get_score,
            scorer=fuzz.token_set_ratio,
            already_ratio=True
        )

    def partial_token_set_ratio(self) -> pd.Series:
        return self._apply_score(
            match_func=self._get_score,
            scorer=fuzz.partial_token_set_ratio,
            already_ratio=True
        )

    def partial_token_sort_ratio(self) -> pd.Series:
        return self._apply_score(
            match_func=self._get_score,
            scorer=fuzz.partial_token_sort_ratio,
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
        }

        return func_mapper[func_name]

    def compare(
        self, 
        scorers: EditDistanceScorers=None, 
        weights: list[int | float]=None,
        drop_intermediate: bool=True
    ) -> pd.Series | pd.DataFrame:

        # Define scorers and weights
        if weights is None and scorers is None:
            default_scorers = {
                "damerau_levenshtein": 1, 
                "jaro": .9, 
                "jaro_winkler": 1, 
                "levenshtein": .9,
                "partial_ratio": .5,
                "token_sort_ratio": .6,
            }
            scorers = default_scorers.keys()
            weights = default_scorers.values()
        elif weights is None and scorers is not None:
            weights = [1 for _ in scorers]

        if len(weights) != len(scorers):
            raise ValueError("Number of scorers and weights must match.")

        sum_w = sum(weights)
        weights = [w / sum_w for w in weights] # normalize to 1
        scorer_dict = {s: w for s, w in zip(scorers, weights)}
        
        scores = dict()
        for func_nm, weight in scorer_dict.items():
            func = self._dispatcher(func_nm)
            scores[f"{func_nm}_score"] = func() * weight

        df = pd.DataFrame(scores)
        df["final_score"] = df[list(scores.keys())].sum(axis=1, skipna=True)

        if drop_intermediate:
            return df["final_score"]

        return df

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

    def _get_matches_distance(
        self, 
        o_str: str, 
        lookup: list[str], 
        scorer: Callable, 
        k_matches: int
    ) -> list[tuple]:
        o_str_len = len(o_str)
        matches = [_get_score_from_distance(lu_str, o_str, o_str_len, scorer) for lu_str in lookup]
        matches = heapq.nlargest(k_matches, matches)

        return matches

    def _get_matches_pct(
        self, 
        o_str: str, 
        lookup: list[str], 
        scorer: Callable, 
        k_matches: int
    ) -> list[tuple]:
        matches = [_get_score(lu_str, o_str, scorer) for lu_str in lookup]
        matches = heapq.nlargest(k_matches, matches)

        return matches

    def _get_all_matches(
        self,
        match_func: Callable,
        scorer: Callable,
        k_matches: int, 
        threshold: int,
        already_ratio: bool=False,
        ncpus: int=None
    ) -> pd.DataFrame:
        original = self._original.to_list() # faster to iterate over list
        lookup = self._lookup.to_list() 
        
        og_colname = self._original.name
        lu_colname = self._lookup.name

        merged = {og_colname: [], lu_colname: [], "score": []}

        with tqdm.tqdm(total=len(self._original.index)) as pbar:
            if ncpus == 1: # this is just single-threading
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
                        pool.submit(match_func, o_str, lookup, scorer, k_matches) \
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
        k_matches: int=5, 
        threshold: int=80, 
        ncpus: int=None,
        ascii_only: bool=False
    ) -> pd.DataFrame:
        if ascii_only:
            scorer = fast_distance.damerau_levenshtein_distance
        else:
            scorer = jellyfish.damerau_levenshtein_distance

        return self._get_all_matches(
            match_func=self._get_matches_distance,
            scorer=scorer,
            k_matches=k_matches, 
            threshold=threshold,
            ncpus=ncpus
        )

    def levenshtein(self, k_matches: int=5, threshold: int=80, ncpus: int=None) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_distance,
            scorer=Levenshtein.distance,
            k_matches=k_matches, 
            threshold=threshold,
            ncpus=ncpus
        )

    def hamming(
        self, 
        k_matches: int=5, 
        threshold: int=80, 
        ncpus: int=None, 
        ascii_only: bool=True
    ) -> pd.DataFrame:
        if ascii_only:
            scorer = fast_distance.hamming_distance
        else:
            scorer = jellyfish.hamming_distance

        return self._get_all_matches(
            match_func=self._get_matches_distance,
            scorer=scorer,
            k_matches=k_matches,
            threshold=threshold,
            ncpus=ncpus
        )

    def jaro_winkler(self, k_matches: int=5, threshold: int=80, ncpus: int=None) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=jarowinkler.jarowinkler_similarity,
            k_matches=k_matches,
            threshold=threshold,
            ncpus=ncpus
        )

    def jaro(self, k_matches: int=5, threshold: int=80, ncpus: int=None) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=jarowinkler.jaro_similarity,
            k_matches=k_matches,
            threshold=threshold,
            ncpus=ncpus
        )

    def ratio(self, k_matches: int=5, threshold: int=80, ncpus: int=None) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=fuzz.ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True,
            ncpus=ncpus
        )

    def partial_ratio(self, k_matches: int=5, threshold: int=80, ncpus: int=None) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=fuzz.partial_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True,
            ncpus=ncpus
        )

    def token_sort_ratio(self, k_matches: int=5, threshold: int=80, ncpus: int=None) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=fuzz.token_sort_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True,
            ncpus=ncpus
        )

    def token_set_ratio(self, k_matches: int=5, threshold: int=80, ncpus: int=None) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=fuzz.token_set_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True,
            ncpus=ncpus
        )

    def partial_token_set_ratio(self, k_matches: int=5, threshold: int=80, ncpus: int=None) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=fuzz.partial_token_set_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True,
            ncpus=ncpus
        )

    def partial_token_sort_ratio(self, k_matches: int=5, threshold: int=80, ncpus: int=None) -> pd.DataFrame:
        return self._get_all_matches(
            match_func=self._get_matches_pct,
            scorer=fuzz.partial_token_sort_ratio,
            k_matches=k_matches,
            threshold=threshold,
            already_ratio=True,
            ncpus=ncpus
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
        drop_intermediate: bool=True,
        ncpus: int=None,
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
            default_scorers = {
                "damerau_levenshtein": 1, 
                "jaro": .9, 
                "jaro_winkler": 1, 
                "levenshtein": .9,
                "tfidf": .8
            }
            scorers = default_scorers.keys()
            weights = default_scorers.values()
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
                df = func(k_matches=k_matches, threshold=0, ncpus=ncpus)
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
