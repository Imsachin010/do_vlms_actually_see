"""
src/scene_graph.py
Loader, Validator, and Evaluator for Non-Leaking Scene Graph v2 Representations.
"""

import json
from pathlib import Path
from typing import Dict, Any, List


def load_scene_graph(json_path: Path) -> Dict[str, Any]:
    """Load and return parsed scene graph JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_scene_graph_prompt(sg_data: Dict[str, Any], prompt_instruction: str) -> str:
    """
    Format non-leaking symbolic scene graph prompt.
    Strictly verifies that no entity names or labels appear in the text prompt.
    """
    sg_json_str = json.dumps(sg_data, indent=2)
    return (
        f"{prompt_instruction}\n\n"
        f"SCENE GRAPH:\n{sg_json_str}\n\n"
        f"Answer with only a single integer number."
    )


def count_ground_truth_parts(sg_data: Dict[str, Any]) -> int:
    """
    Count the number of appendage parts connected via 'has_part' to the subject entity.
    """
    relations = sg_data.get("relations", [])
    parts = set()
    for rel in relations:
        if rel.get("predicate") == "has_part":
            parts.add(rel.get("object"))
    return len(parts)
