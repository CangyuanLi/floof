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
    "cosine",
    "cosine_ascii",
    "bag",
    "bag_ascii",
    "overlap",
    "overlap_ascii",
    "tversky",
    "tversky_ascii",
    "levenshtein",
    "levenshtein_ascii",
    "damerau_levenshtein",
    "damerau_levenshtein_ascii",
    "osa",
    "osa_ascii",
    "soundex",
    "soundex_ascii",
]

RustSliceScorers = Literal[
    "hamming_similarity",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "levenshtein_similarity",
    "damerau_levenshtein_similarity",
    "damerau_levenshtein_similarity_ascii",
    "osa_similarity",
    "jaccard_similarity",
    "sorensen_dice_similarity",
    "cosine_similarity",
    "bag_similarity",
    "overlap_similarity",
    "tversky_similarity",
    "soundex_similarity",
]

ProcessedUnicode = list[str]
ProcessedAscii = list[int]

def hamming(s1: str, s2: str, ascii_only: bool = False) -> float:
    """Calculates the extend Hamming similarity between two strings. The Hamming
    distance is defined as the total number of differences.

    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        Tells floof the string contains only ASCII characters (floof will not do ANY
        validation), allowing for certain optimizations, by default False

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
        Tells floof the string contains only ASCII characters (floof will not do ANY
        validation), allowing for certain optimizations, by default False

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
        Tells floof the string contains only ASCII characters (floof will not do ANY
        validation), allowing for certain optimizations, by default False

    Returns
    -------
    float
    """
    ...

def cosine(s1: str, s2: str, ascii_only: bool = False) -> float:
    r"""Calculates the Otsuka-Ochiai coefficient between two strings. Each string is
    broken down into a set of characters, making it analogous to the Cosine distance
    between two vectors. It is calculated as:

    ..math::

        K = \frac{|A \cap B|}{\sqrt{|A| \times |B|}}

    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        Tells floof the string contains only ASCII characters (floof will not do ANY
        validation), allowing for certain optimizations, by default False

    Returns
    -------
    float
    """
    ...

def bag(s1: str, s2: str, ascii_only: bool = False) -> float:
    r"""Calculates the bag similarity between two strings. A bag is a set that contains
    each character and its frequency in the string. Then,

    ..math::

        B(A, B) = max(|Bag(s1) - Bag(s2)|, |Bag(s2) - Bag(s2)|)

    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        Tells floof the string contains only ASCII characters (floof will not do ANY
        validation), allowing for certain optimizations, by default False

    Returns
    -------
    float
    """
    ...

def overlap(s1: str, s2: str, ascii_only: bool = False) -> float:
    r"""Calculates the overlap coefficient between two sets. A string's characters are
    turned into a set. It is calculated as:

    ..math::

        O(A, B) = |A \cap B| / min(|A|, |B|)

    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        Tells floof the string contains only ASCII characters (floof will not do ANY
        validation), allowing for certain optimizations, by default False

    Returns
    -------
    float
    """
    ...

def tversky(s1: str, s2: str, ascii_only: bool = False) -> float:
    r"""Calculates the Tversky Index between two sets. A string's characters are
    turned into a set. It is calculated as:

    ..math::

        T(A, B) = \frac{|A \cap B|}{|A \cap B| + \alpha |A - B| + \beta |B - A|}

    where `alpha` and `beta` are set to 0.5.

    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        Tells floof the string contains only ASCII characters (floof will not do ANY
        validation), allowing for certain optimizations, by default False

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
        Tells floof the string contains only ASCII characters (floof will not do ANY
        validation), allowing for certain optimizations, by default False

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
        Tells floof the string contains only ASCII characters (floof will not do ANY
        validation), allowing for certain optimizations, by default False

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
        Tells floof the string contains only ASCII characters (floof will not do ANY
        validation), allowing for certain optimizations, by default False

    Returns
    -------
    float
    """

def damerau_levenshtein(s1: str, s2: str, ascii_only: bool = False) -> float:
    """Calculates the Damerau-Levenshtein distance between two strings and scales it by
    the length of the longest string. The Damerau-Levenshtein algorithm allows for four
    operations: substitutions (e.g. "foo" -> "fou"), insertions (e.g. "ba" -> "bar"),
    deletions (e.g. "foo" -> "fo"), and transpositions (e.g. "bay" -> "bya"). The
    distance is the mininum number of operations needed to transform one string into
    the other.

    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        Tells floof the string contains only ASCII characters (floof will not do ANY
        validation), allowing for certain optimizations, by default False

    Returns
    -------
    float
    """

def osa(s1: str, s2: str, ascii_only: bool = False) -> float:
    """Calculates the Optimal String Alignment distance between two strings and scales
    it by the length of the longest string. It is also called the Restricted Levenshtein
    Distance. It allows for three operations: substitutions (e.g. "foo" -> "fou"),
    insertions (e.g. "ba" -> "bar"), and deletions (e.g. "foo" -> "fo"). The
    distance is the mininum number of operations needed to transform one string into
    the other.

    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        Tells floof the string contains only ASCII characters (floof will not do ANY
        validation), allowing for certain optimizations, by default False

    Returns
    -------
    float
    """

def soundex(s1: str, s2: str, ascii_only: bool = False) -> float:
    """Calculates the American Soundex of the two strings and returns the Hamming
    similarity between the two encodings. Note that Soundex is only defined for ASCII
    strings, and the default is to use the `deunicode` crate to make a best-effort map
    between Unicode and ASCII.

    Parameters
    ----------
    s1 : str
    s2 : str
    ascii_only : bool, optional
        Tells floof the string contains only ASCII characters (floof will not do ANY
        validation), allowing for certain optimizations, by default False

    Returns
    -------
    float
    """
    ...

def _extract_graphemes(arr1: list[str]) -> list[ProcessedUnicode]: ...
def _extract_bytes(arr1: list[str]) -> list[ProcessedAscii]: ...
def _extract_graphemes_tup(arr1: list[str]) -> list[tuple[str, ProcessedUnicode]]: ...
def _extract_bytes_tup(arr1: list[str]) -> list[tuple[str, ProcessedAscii]]: ...
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
