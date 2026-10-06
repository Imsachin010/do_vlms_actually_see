"""
src/parse.py
Single Source of Truth for Model Output Extraction and Error Classification.
"""

import re
from pathlib import Path
from typing import Optional, Tuple

WORD_TO_INT = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12
}


def parse_integer(text: Optional[str]) -> Optional[int]:
    """
    Extract a single integer count from a model's free-form text response.
    Prioritizes leading numbers, isolated tokens, and numeric word forms.
    """
    if text is None:
        return None
    raw = str(text).strip().lower()
    if not raw:
        return None

    # 1. Direct regex for standalone integer or decimal (e.g. '3', '4.', '5 legs')
    m = re.search(r"\b(\d{1,2})\b", raw)
    if m:
        return int(m.group(1))

    # 2. Number words
    for word, val in WORD_TO_INT.items():
        if re.search(rf"\b{word}\b", raw):
            return val

    # 3. Trailing/leading digits
    m_any = re.search(r"(\d+)", raw)
    if m_any:
        return int(m_any.group(1))

    return None


def classify_error(
    predicted: Optional[int],
    true_legs: int,
    canonical_legs: int = 4
) -> str:
    """
    Strict 4-category taxonomy of VLM outputs on counterfactual queries:
    - 'correct': predicted == true_legs
    - 'prior-aligned': predicted == canonical_legs (4) when true_legs != 4
    - 'off-by-one': abs(predicted - true_legs) == 1 and predicted != canonical_legs
    - 'visual-confusion': other plausible miscounts (e.g. 2 or 6)
    - 'other': unparseable, null, or extreme outlier
    """
    if predicted is None or predicted < 0:
        return "other"

    if predicted == true_legs:
        return "correct"

    if predicted == canonical_legs and true_legs != canonical_legs:
        return "prior-aligned"

    if abs(predicted - true_legs) == 1:
        return "off-by-one"

    if predicted in {0, 1, 2, 6, 7, 8}:
        return "visual-confusion"

    return "other"


def parse_filename(filename: str) -> Tuple[str, Optional[int], Optional[str]]:
    """
    Parse concept, true_legs, and background from standard image filenames.
    Format: '<concept>_<N>legs_<background>.<ext>'
    """
    stem = Path(filename).stem
    m = re.match(r"^([a-zA-Z]+)_(\d+)legs_([a-zA-Z]+)$", stem)
    if m:
        return m.group(1).lower(), int(m.group(2)), m.group(3).lower()
    return stem, None, None
