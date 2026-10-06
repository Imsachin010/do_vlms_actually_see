import base64
import requests
import time


class GroqVLM:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def query(self, image_path, prompt):
        image_b64 = self.encode_image(image_path)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url":{"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 20,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start = time.time()

        response = requests.post(self.url, json=payload, headers=headers)
        latency = time.time() - start

        if response.status_code != 200:
            from src.utils.logger import setup_logger
            _log = setup_logger()
            _log.warning(f"Groq API error [{response.status_code}]: {response.text}")
            return None, latency

        result = response.json()
        text = result["choices"][0]["message"]["content"]

        return text.strip(), latency