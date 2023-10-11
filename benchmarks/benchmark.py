# Imports
import random
import string
from pathlib import Path

import polars as pl

import floof

DAT_PATH = Path(__file__).resolve().parents[1] / "test_data"

UNICODE_INCLUDE_RANGES = [
    (0x0021, 0x0021),
    (0x0023, 0x0026),
    (0x0028, 0x007E),
    (0x00A1, 0x00AC),
    (0x00AE, 0x00FF),
    (0x0100, 0x017F),
    (0x0180, 0x024F),
    (0x2C60, 0x2C7F),
    (0x16A0, 0x16F0),
    (0x0370, 0x0377),
    (0x037A, 0x037E),
    (0x0384, 0x038A),
    (0x038C, 0x038C),
]

ALPHABET = [
    chr(code_point)
    for current_range in UNICODE_INCLUDE_RANGES
    for code_point in range(current_range[0], current_range[1] + 1)
]


def random_unicode_string(length: int) -> str:
    # Update this to include code point ranges to be sampled
    return "".join(random.choices(ALPHABET, k=length))


def random_ascii_string(length: int) -> str:
    return "".join(random.choices(string.printable, k=length))


def create_unicode_db():
    col1 = [random_unicode_string(random.randint(0, 100)) for _ in range(1_000_000)]
    col2 = [random_unicode_string(random.randint(0, 100)) for _ in range(1_000_000)]

    pl.DataFrame(data={"col1": col1, "col2": col2}).write_parquet(
        DAT_PATH / "unicode.parquet"
    )


def create_ascii_db():
    col1 = [random_ascii_string(random.randint(0, 100)) for _ in range(1_000_000)]
    col2 = [random_ascii_string(random.randint(0, 100)) for _ in range(1_000_000)]

    pl.DataFrame(data={"col1": col1, "col2": col2}).write_parquet(
        DAT_PATH / "ascii.parquet"
    )


def main():
    df = pl.scan_parquet(DAT_PATH / "ascii.parquet").fetch(100_000)
    matcher_unicode = floof.Matcher(df.get_column("col1"), df.get_column("col2"))
    matcher_ascii = floof.Matcher(
        df.get_column("col1"), df.get_column("col2"), ascii_only=True
    )

    # matcher_unicode._get_all_matches_rust("hamming_ascii", k_matches=5, threshold=0)
    # matcher_unicode._get_all_matches_rust("jaro_ascii", k_matches=5, threshold=0)
    # matcher_ascii.jaro()
    matcher_ascii.hamming()


if __name__ == "__main__":
    main()
