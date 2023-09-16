from typing import Literal

RustScorers = Literal["hamming", "hamming_ascii", "jaccard", "jaccard_ascii"]

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
        The Hamming distance scaled by the lengths of the strings
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
        The Jaccard similarity
    """
    ...

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
