import os
import requests
from urllib.parse import quote
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def download_images(query, save_dir, num_images=7):
    os.makedirs(save_dir, exist_ok=True)

    search_url = f"https://www.bing.com/images/search?q={quote(query)}&form=HDRSC2"

    response = requests.get(search_url, headers=HEADERS)
    html = response.text

    links = []
    for part in html.split("murl&quot;:&quot;")[1:]:
        link = part.split("&quot;")[0]
        links.append(link)

    count = 0

    for i, link in enumerate(links):
        if count >= num_images:
            break

        try:
            img_data = requests.get(link, timeout=5).content
            with open(f"{save_dir}/{query}_{i}.jpg", "wb") as f:
                f.write(img_data)

            count += 1

        except:
            continue

    print(f"{query}: Downloaded {count} images")


def main():
    concepts = [
        "Dog full body",
        "Cat full body",
        "Lion full body",
        "Elephant full body",
        "Horse full body",
        "Cheetah full body",
        "Leopard full body",
        "Fox full body",
        "Deer full body",
        "Hyena full body",
        "Serval animal full body",
        "Binturong animal full body",
        "Numbat animal full body",
        "Fossa animal full body",
        "Okapi animal full body",
    ]

    base_path = "data/raw/canonical_images"

    for concept in concepts:
        folder = concept.split()[0]  # Dog, Cat, etc.
        save_dir = os.path.join(base_path, folder)

        download_images(concept, save_dir)


if __name__ == "__main__":
    main()