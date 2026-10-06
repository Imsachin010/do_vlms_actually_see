import requests
import base64
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_path = "data/raw/counterfactual_images/cat_3legs_natural.png"
base64_str = encode_image(image_path)

payload = {
    "model": "moondream",
    "messages": [
        {
            "role": "user",
            "content": "How many legs does this animal exactly have in this image? Answer with just a single number integer, nothing else.",
            "images": [base64_str]
        }
    ],
    "stream": False,
    "options": {
        "temperature": 0.0
    }
}

try:
    response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=60)
    print(f"Status Code: {response.status_code}")
    print("Exact content string returned by Moondream:")
    txt = response.json().get("message", {}).get("content", "")
    print(f"'{txt}'")
except Exception as e:
    print(f"Exception: {e}")
