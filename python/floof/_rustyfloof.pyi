from typing import Literal

RustScorers = Literal["hamming", "hamming_ascii", "jaccard", "jaccard_ascii"]

def hamming(s1: str, s2: str, ascii_only: bool = False) -> float: ...
def jaccard(s1: str, s2: str, ascii_only: bool = False) -> float: ...
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
