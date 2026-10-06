"""
src/schema.py
Fixed Schema Definition and Validator for Evaluation Result CSVs.
Schema: [model, condition, backend, concept, true_legs, background, raw_output, parsed_int, error_class]
"""

from typing import List, Dict, Any
import pandas as pd
from src.parse import classify_error, parse_integer

EXPECTED_COLUMNS: List[str] = [
    "model",
    "condition",
    "backend",
    "concept",
    "true_legs",
    "background",
    "raw_output",
    "parsed_int",
    "error_class"
]


def format_evaluation_row(
    model: str,
    condition: str,
    backend: str,
    concept: str,
    true_legs: int,
    background: str,
    raw_output: str,
    parsed_int: int | None = None
) -> Dict[str, Any]:
    """
    Format a single query evaluation record according to the standard schema.
    """
    if parsed_int is None:
        parsed_int = parse_integer(raw_output)

    error_class = classify_error(parsed_int, true_legs)

    return {
        "model": str(model).strip(),
        "condition": str(condition).strip().lower(),
        "backend": str(backend).strip(),
        "concept": str(concept).strip().lower(),
        "true_legs": int(true_legs),
        "background": str(background).strip().lower(),
        "raw_output": str(raw_output).strip(),
        "parsed_int": int(parsed_int) if parsed_int is not None else -1,
        "error_class": error_class
    }


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Ensure the DataFrame conforms to the fixed CSV schema.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required schema columns: {missing}")
    return True
