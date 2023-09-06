from __future__ import annotations

from typing import Literal

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

NearestNeighborScorers = Literal["tfidf",]

PhoneticScorers = Literal[
    "match_rating_codex",
    "metaphone",
    "nysiis",
    "soundex",
]

AllScorers = Literal[EditDistanceScorers, NearestNeighborScorers, PhoneticScorers]
