import re
from collections.abc import Callable


def _get_score(o_str: str, lu_str: str, scorer: Callable) -> tuple[float, str, str]:
    if o_str == "" or lu_str == "":
        return (0, o_str, lu_str)

    pct = scorer(o_str, lu_str)

    return (pct, o_str, lu_str)


def _get_score_from_distance(
    o_str: str, lu_str: str, scorer: Callable
) -> tuple[float, str, str]:
    if o_str == "" or lu_str == "":
        return (0, o_str, lu_str)

    dist = scorer(o_str, lu_str)

    # For methods that return a distance instead of a percentage, divide the returned
    # distance by the length of the longest string to obtain a relative percentage.
    # Otherwise, the distance when comparing long string to short strings will be biased
    # against long srings.
    max_len = max(len(o_str), len(lu_str))
    pct = (max_len - dist) / max_len

    return (pct, o_str, lu_str)


def _get_score_phonetic(
    o_str: str, lu_str: str, scorer: Callable
) -> tuple[float, str, str]:
    return (scorer(o_str) == scorer(lu_str)) * 100


def _normalize(lst: list[float]) -> list[float]:
    sum_ = sum(lst)

    return [x / sum_ for x in lst]


def _get_ngrams(string: str, n: int = 3, strip_punc: bool = True) -> list[str]:
    if strip_punc:
        pattern = re.compile(r"[,-./]|\sBD")
        string = re.sub(pattern, "", string)

    ngrams = zip(*[string[i:] for i in range(n)])

    return ["".join(ngram) for ngram in ngrams]
