from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Union

import numpy as np
import pandas as pd
import polars as pl

ArrayLike = Union[Sequence, pd.Series, pl.Series, np.ndarray]

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
