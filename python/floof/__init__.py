__version__ = "0.1.5"

from ._comparer import Comparer
from ._matcher import Matcher
from ._rustyfloof import (
    cosine,
    damerau_levenshtein,
    hamming,
    jaccard,
    jaro,
    jaro_winkler,
    levenshtein,
    sorensen_dice,
)
