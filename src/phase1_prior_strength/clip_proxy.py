import torch
import clip
import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_images(base_path="data/raw/canonical_images"):
    data = {}

    for concept in os.listdir(base_path):
        concept_path = os.path.join(base_path, concept)

        if not os.path.isdir(concept_path):
            continue

        images = []
        for file in os.listdir(concept_path):
            if file.endswith((".png", ".jpg", ".jpeg", ".avif")):
                images.append(os.path.join(concept_path, file))

        data[concept] = images

    return data


def compute_embeddings(image_paths, model, preprocess):
    embeddings = []

    for img_path in image_paths:
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = model.encode_image(image)

            embeddings.append(emb.cpu().numpy()[0])

        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")

    return np.array(embeddings)

from sklearn.metrics.pairwise import cosine_similarity

def compute_concentration(embeddings):
    if len(embeddings) < 2:
        return 0

    sim_matrix = cosine_similarity(embeddings)

    # remove diagonal (self similarity)
    n = sim_matrix.shape[0]
    sim_sum = (sim_matrix.sum() - n)  # remove diagonal ones
    count = n * (n - 1)

    return sim_sum / count



def run():
    logger.info("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    logger.info("Loading images...")
    data = load_images()

    results = []

    for concept, image_paths in data.items():
        logger.info(f"Processing {concept} ({len(image_paths)} images)")

        embeddings = compute_embeddings(image_paths, model, preprocess)

        concentration = compute_concentration(embeddings)

        results.append({
            "concept": concept,
            "clip_concentration": concentration
        })

    df = pd.DataFrame(results)

    # Normalize
    df["clip_norm"] = (df["clip_concentration"] - df["clip_concentration"].min()) / (
        df["clip_concentration"].max() - df["clip_concentration"].min() + 1e-8
    )
    # Add better version
    df["clip_zscore"] = (
        df["clip_concentration"] - df["clip_concentration"].mean()
    ) / (df["clip_concentration"].std() + 1e-8)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/clip_scores.csv", index=False)

    logger.info("Saved CLIP scores")


if __name__ == "__main__":
    run()