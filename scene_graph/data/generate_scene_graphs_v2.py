"""
Generate scene graphs v2 for all 60 counterfactual images.
Redesigned to require true counting inference (Weakness 1 fix).

Schema rules:
1. No `true_legs` field in scene graph JSON.
2. No direct count relations (e.g. "has 5 legs") or ordinals ("fifth").
3. Anatomically neutral, position-based leg relations.
4. 3 legs: omit one leg (absence inferred).
5. 5 legs: add "additional hind leg" or "extra front leg" (non-numeric descriptor).
"""

import json
from pathlib import Path

CONCEPTS = [
    "dog", "cat", "lion", "elephant", "horse",
    "cheetah", "leopard", "fox", "deer", "hyena",
    "serval", "binturong", "numbat", "fossa", "okapi"
]

LEG_COUNTS = [3, 5]
BACKGROUNDS = ["plain", "natural"]

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


def build_scene_graph_v2(concept: str, legs: int, bg: str) -> tuple[dict, dict]:
    """
    Returns (scene_graph_dict, annotation_dict)
    scene_graph_dict MUST NOT contain `true_legs` or direct leg count strings.
    """
    attrs = DEFAULT_ATTRIBUTES.get(concept, {"color": "unknown", "size": "medium"}).copy()
    image_id = f"{concept}_{legs}legs_{bg}"

    # Build anatomical position relations
    if legs == 3:
        leg_relations = [
            f"{concept} has a front-left leg",
            f"{concept} has a front-right leg",
            f"{concept} has a hind-left leg",
        ]
    elif legs == 5:
        leg_relations = [
            f"{concept} has a front-left leg",
            f"{concept} has a front-right leg",
            f"{concept} has a hind-left leg",
            f"{concept} has a hind-right leg",
            f"{concept} has an additional hind leg",
        ]
    else:
        raise ValueError(f"Unsupported leg count: {legs}")

    bg_relation = (
        f"{concept} stands on plain white background"
        if bg == "plain"
        else f"{concept} stands in natural habitat"
    )

    all_relations = leg_relations + [bg_relation]

    # Validate relation count sum before constructing final dict
    leg_rel_count = sum(1 for r in all_relations if "leg" in r)
    assert leg_rel_count == legs, f"Assertion failed for {image_id}: count {leg_rel_count} != ground truth {legs}"

    sg_dict = {
        "image_id": image_id,
        "concept": concept,
        "background": bg,
        "objects": [concept],
        "attributes": {
            concept: attrs,
        },
        "relations": all_relations,
    }

    ann_dict = {
        "image_id": image_id,
        "concept": concept,
        "true_legs": legs,
        "background": bg,
        "question": "Based solely on the relations listed in the following scene description, count how many distinct legs are described.",
        "ground_truth": str(legs),
    }

    return sg_dict, ann_dict


def run():
    out_dir = Path("scene_graph/data/scene_graphs_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_sgs = []
    all_anns = []

    for concept in CONCEPTS:
        for legs in LEG_COUNTS:
            for bg in BACKGROUNDS:
                sg, ann = build_scene_graph_v2(concept, legs, bg)
                all_sgs.append(sg)
                all_anns.append(ann)

                # Save individual JSON
                fname = f"{sg['image_id']}.json"
                with open(out_dir / fname, "w", encoding="utf-8") as f:
                    json.dump(sg, f, indent=2)

    # Master scene graphs JSON
    master_sg_path = Path("scene_graph/data/scene_graphs_v2.json")
    with open(master_sg_path, "w", encoding="utf-8") as f:
        json.dump(all_sgs, f, indent=2)

    # Master annotations JSON
    ann_path = Path("scene_graph/data/annotations_v2.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(all_anns, f, indent=2)

    print(f"[OK] Generated {len(all_sgs)} scene graphs v2 at {master_sg_path}")
    print(f"[OK] Generated {len(all_anns)} annotations v2 at {ann_path}")


if __name__ == "__main__":
    run()
