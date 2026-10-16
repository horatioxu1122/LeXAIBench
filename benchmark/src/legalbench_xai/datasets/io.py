import csv
import re
from pathlib import Path
from typing import Any

# Exponent-notation numbers such as "8e-05", "1E5" or "-3.2e+1". v1 parsed only
# plain integers and numbers containing a ".", so an exponent form without a "."
# (Give Me Some Credit has values like "8e-05") stayed a string and became a
# spurious category. Words such as "inf" or "nan" do not match this pattern.
_EXPONENT_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+")


def _coerce_scalar(value: str, *, parse_exponent_notation: bool = False) -> Any:
    """Convert one CSV cell to None, bool, int, float or str.

    parse_exponent_notation=False is the v1 behavior of the published bundles.
    True (cslaw corrections ``numeric_parsing_fix``) also parses exponent-notation
    numbers that v1 left as strings; every other input converts as in v1.
    """
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"na", "n/a", "nan", "null", "none", "?"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        if parse_exponent_notation and _EXPONENT_NUMBER.fullmatch(text):
            return float(text)
        return text


def load_csv_rows(
    path: Path,
    *,
    numeric_parsing_fix: bool = False,
    text_columns: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Rows of a CSV file. Columns in text_columns keep their text (missing cells become None), so codes such
    as "09190" stay categories; every other cell converts with _coerce_scalar."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    key: (_text_or_none(value) if key in text_columns
                          else _coerce_scalar(value, parse_exponent_notation=numeric_parsing_fix))
                    for key, value in row.items()
                }
            )
    return rows


def _text_or_none(value: str | None) -> str | None:
    text = (value or "").strip()
    return None if text == "" or text.lower() in {"na", "n/a", "nan", "null", "none", "?"} else text


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []
