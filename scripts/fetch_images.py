"""
fetch_images.py
Downloads the 60 counterfactual quadruped images from the Hugging Face Dataset repo.
Usage:
    python scripts/fetch_images.py --dest data/counterfactual_images
"""

import argparse
import os
import urllib.request
from pathlib import Path
import pandas as pd

HF_REPO_ID = "Imsachin010/vlm-bias-counterfactual-images"  # Published HF dataset repo
BASE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/images"

def fetch_images(dest_dir: str):
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("data/image_manifest.csv")
    if not manifest_path.exists():
        manifest_path = Path("hf_dataset_bundle/image_manifest.csv")

    if not manifest_path.exists():
        print(f"[ERROR] Manifest file not found at {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    print(f"Fetching {len(df)} images into {dest} ...")

    # If local bundle exists, copy directly
    local_bundle = Path("hf_dataset_bundle/images")
    if local_bundle.exists() and len(list(local_bundle.glob("*.*"))) >= 60:
        print("[INFO] Found local image bundle. Copying locally without network latency...")
        for _, row in df.iterrows():
            fname = row["image_filename"]
            src = local_bundle / fname
            dst = dest / fname
            if src.exists() and not dst.exists():
                import shutil
                shutil.copy2(src, dst)
        print(f"[OK] Successfully populated {dest} with {len(df)} images.")
        return

    # Otherwise download from Hugging Face
    success = 0
    for idx, row in df.iterrows():
        fname = row["image_filename"]
        target = dest / fname
        if target.exists():
            success += 1
            continue

        url = f"{BASE_URL}/{fname}"
        try:
            urllib.request.urlretrieve(url, target)
            success += 1
            print(f"  [{idx+1:2d}/{len(df)}] Downloaded {fname}")
        except Exception as e:
            print(f"  [WARN] Failed to download {fname} from {url}: {e}")

    print(f"[FINISHED] Successfully downloaded {success}/{len(df)} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download counterfactual stimulus images.")
    parser.add_argument("--dest", type=str, default="data/counterfactual_images", help="Target directory")
    args = parser.parse_args()
    fetch_images(args.dest)
