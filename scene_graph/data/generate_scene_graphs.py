"""
Generate scene graphs for all counterfactual images.
Each scene graph is a structured JSON describing the scene in a fixed format,
stripping all visual/perceptual ambiguity.

Standard format:
{
  "image_id": "dog_3legs_plain",
  "concept": "dog",
  "true_legs": 3,
  "background": "plain",
  "objects": ["dog"],
  "attributes": {
    "dog": {
      "legs": 3,
      "color": "brown",
      "size": "medium"
    }
  },
  "relations": [
    "dog has 3 legs",
    "dog stands on plain white background"
  ]
}
"""

import json
import os
from pathlib import Path

# ── mirror the 15 concepts from the main experiment ──────────────────────
CONCEPTS = [
    "dog", "cat", "lion", "elephant", "horse",
    "cheetah", "leopard", "fox", "deer", "hyena",
    "serval", "binturong", "numbat", "fossa", "okapi"
]

LEG_COUNTS = [3, 5]
BACKGROUNDS = ["plain", "natural"]

# Descriptive attributes per concept (kept minimal and consistent)
DEFAULT_ATTRIBUTES = {
    "dog":       {"color": "brown",  "size": "medium"},
    "cat":       {"color": "orange", "size": "small"},
    "lion":      {"color": "golden", "size": "large"},
    "elephant":  {"color": "gray",   "size": "very large"},
    "horse":     {"color": "brown",  "size": "large"},
    "cheetah":   {"color": "yellow", "size": "medium"},
    "leopard":   {"color": "spotted","size": "medium"},
    "fox":       {"color": "red",    "size": "small"},
    "deer":      {"color": "brown",  "size": "medium"},
    "hyena":     {"color": "spotted","size": "medium"},
    "serval":    {"color": "golden", "size": "small"},
    "binturong": {"color": "black",  "size": "medium"},
    "numbat":    {"color": "red-brown","size": "small"},
    "fossa":     {"color": "brown",  "size": "medium"},
    "okapi":     {"color": "brown-striped","size": "large"},
}


def build_scene_graph(concept: str, legs: int, bg: str) -> dict:
    attrs = DEFAULT_ATTRIBUTES.get(concept, {"color": "unknown", "size": "medium"}).copy()
    attrs["legs"] = legs

    image_id = f"{concept}_{legs}legs_{bg}"

    if bg == "plain":
        rels = [
            f"{concept} has {legs} legs",
            f"{concept} stands on plain white background",
        ]
    else:
        rels = [
            f"{concept} has {legs} legs",
            f"{concept} stands in natural habitat",
        ]

    return {
        "image_id": image_id,
        "concept": concept,
        "true_legs": legs,
        "background": bg,
        "objects": [concept],
        "attributes": {
            concept: attrs,
        },
        "relations": rels,
    }


def build_annotations_json(sg_list: list[dict]) -> list[dict]:
    """Build the ground-truth annotations file for evaluation."""
    anns = []
    for sg in sg_list:
        anns.append({
            "image_id": sg["image_id"],
            "concept": sg["concept"],
            "true_legs": sg["true_legs"],
            "background": sg["background"],
            "question": "How many legs does this animal have?",
            "ground_truth": str(sg["true_legs"]),
        })
    return anns


def run():
    out_dir = Path("scene_graph/data/scene_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_sgs = []

    for concept in CONCEPTS:
        for legs in LEG_COUNTS:
            for bg in BACKGROUNDS:
                sg = build_scene_graph(concept, legs, bg)
                all_sgs.append(sg)

                # write individual per-image scene graph
                fname = f"{sg['image_id']}.json"
                with open(out_dir / fname, "w") as f:
                    json.dump(sg, f, indent=2)

    # ── write master scene-graphs JSON (for batch loading) ────────────────
    master_path = Path("scene_graph/data/scene_graphs.json")
    with open(master_path, "w") as f:
        json.dump(all_sgs, f, indent=2)
    print(f"Wrote {len(all_sgs)} scene graphs -> {master_path}")

    # ── write annotations (ground-truth answers) ─────────────────────────
    anns = build_annotations_json(all_sgs)
    ann_path = Path("scene_graph/data/annotations.json")
    with open(ann_path, "w") as f:
        json.dump(anns, f, indent=2)
    print(f"Wrote {len(anns)} annotations     -> {ann_path}")

    # ── write minimal prompts.json for documentation ──────────────────────
    prompts_list = []
    for sg in all_sgs:
        prompts_list.append({
            "image_id": sg["image_id"],
            "concept": sg["concept"],
            "true_legs": sg["true_legs"],
            "background": sg["background"],
        })
    prompts_path = Path("scene_graph/data/prompts.json")
    with open(prompts_path, "w") as f:
        json.dump(prompts_list, f, indent=2)
    print(f"Wrote {len(prompts_list)} prompt-entries -> {prompts_path}")


if __name__ == "__main__":
    run()
