from typing import Literal

RustScorers = Literal[
    "hamming",
    "hamming_ascii",
    "jaccard",
    "jaccard_ascii",
    "jaro",
    "jaro_ascii",
    "sorensen_dice",
    "sorensen_dice_ascii",
    "levenshtein",
    "levenshtein_ascii",
]

RustSliceScorers = Literal[
    "hamming_similarity",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "levenshtein_similarity",
]

ProcessedUnicode = tuple[str, list[str]]
ProcessedAscii = tuple[str, list[int]]

def hamming(s1: str, s2: str, ascii_only: bool = False) -> float:
    """Calculates the extend Hamming similarity between two strings. The Hamming
    distance is defined as the total number of differences.

    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        If the string contains only ASCII characters, avoids the (expensive) operation
        of creating Unicode graphemes, by default False

    Returns
    -------
    float
    """
    ...

def jaccard(s1: str, s2: str, ascii_only: bool = False) -> float:
    r"""Calculates the Jaccard similarity between two strings. The Jaccard similarity
    is a metric for the proximity between two sets, and is given by:

    ..math::

        J(A, B) = \frac{|A \cap B|}{|A \cup B|}

    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        If the string contains only ASCII characters, avoids the (expensive) operation
        of creating Unicode graphemes, by default False

    Returns
    -------
    float
    """
    ...

def sorensen_dice(s1: str, s2: str, ascii_only: bool = False) -> float:
    r"""Calculates the Sorensen-Dice similarity between two strings. It is given by

    ..math::

        S(A, B) = \frac{2J}{1 + J}

    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        If the string contains only ASCII characters, avoids the (expensive) operation
        of creating Unicode graphemes, by default False

    Returns
    -------
    float
    """
    ...

def jaro(s1: str, s2: str, ascii_only: bool = False) -> float:
    r"""Calculates the Jaro similarity between two strings. The Jaro similarity is given
    as:

    ..math::

        \frac{1}{3}(\frac{m}{|s_1|} + \frac{m}{|s2|} + \frac{m - t}{m})

    if m > 0, and 0 otherwise. `s_i` is the length of the string, `m` is the number of
    matches, where a match is if two characters are the same and are within

    ..math::

        [\frac{1}{2}max(|s_1|, |s_2|)] - 1

    characters. `t` is the number of transpositions, or the number of matching
    characters that are not in the right order (divided by two).


    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        If the string contains only ASCII characters, avoids the (expensive) operation
        of creating Unicode graphemes, by default False

    Returns
    -------
    float
    """

def jaro_winkler(s1: str, s2: str, ascii_only: bool = False) -> float:
    r"""Calculates the Jaro-Winkler similarity between two strings. The Jaro-Winkler
    similarity is given as

    ..math::

        jaro_{sim} + lp(1 - jaro_{sim})

    where `jaro_sim` is the Jaro similarity, `l` is the length of the common prefix at
    the start of the string (bounded at 4), and `p` is a constant scaling factor set at
    `0.1`.

    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        If the string contains only ASCII characters, avoids the (expensive) operation
        of creating Unicode graphemes, by default False

    Returns
    -------
    float
    """

def levenshtein(s1: str, s2: str, ascii_only: bool = False) -> float:
    """Calculates the Levenshtein distance between two strings and scales it by the
    length of the longest string. The Levenshtein algorithm allows for two operations:
    substitution (e.g. "foo" -> "fou") and insertion (e.g. "ba" -> "bar"). The distance
    is the mininum number of operations needed to transform one string into the other.

    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        If the string contains only ASCII characters, avoids the (expensive) operation
        of creating Unicode graphemes, by default False

    Returns
    -------
    float
    """

def _extract_graphemes(arr1: list[str]) -> list[ProcessedUnicode]: ...
def _extract_bytes(arr1: list[str]) -> list[ProcessedAscii]: ...
def _compare(
    arr1: list[str], arr2: list[str], func_name: RustScorers, n_jobs: int = 0
) -> list[float]: ...
def _match(
    arr1: list[str],
    arr2: list[str],
    func_name: RustScorers,
    k_matches: int = 5,
    threshold: float = 0,
    n_jobs: int = 0,
    quiet: bool = False,
): ...
def _match_slice(
    arr1: list[ProcessedUnicode],
    arr2: list[ProcessedUnicode],
    func_name: RustSliceScorers,
    k_matches: int = 5,
    threshold: float = 0,
    n_jobs: int = 0,
    quiet: bool = False,
): ...
def _match_slice_ascii(
    arr1: list[ProcessedAscii],
    arr2: list[ProcessedAscii],
    func_name: RustSliceScorers,
    k_matches: int = 5,
    threshold: float = 0,
    n_jobs: int = 0,
    quiet: bool = False,
): ...
