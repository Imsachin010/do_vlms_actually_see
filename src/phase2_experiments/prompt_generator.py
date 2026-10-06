import json
from pathlib import Path

CONCEPTS = [
    "dog","cat","lion","elephant","horse",
    "cheetah","leopard","fox","deer","hyena",
    "serval","binturong","numbat","fossa","okapi"
]

LEG_COUNTS = [3, 5, 6]
BACKGROUNDS = ["plain", "natural"]


def generate_prompt(concept, legs, bg):
    if bg == "plain":
        return f"""
A photorealistic {concept} with exactly {legs} visible legs,
standing on a plain white background.
Full body side view, all {legs} legs clearly visible and countable.
"""
    else:
        return f"""
A photorealistic {concept} with exactly {legs} visible legs
in a natural habitat.
Full body side view, all {legs} legs clearly visible and countable.
"""


def run():
    prompts = []

    for concept in CONCEPTS:
        for legs in LEG_COUNTS:
            for bg in BACKGROUNDS:
                prompt = generate_prompt(concept, legs, bg)

                image_id = f"{concept}_{legs}legs_{bg}"

                prompts.append({
                    "image_id": image_id,
                    "concept": concept,
                    "legs": legs,
                    "background": bg,
                    "prompt": prompt.strip()
                })

    Path("data/metadata").mkdir(parents=True, exist_ok=True)

    with open("data/metadata/image_prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Generated {len(prompts)} prompts")


if __name__ == "__main__":
    run()