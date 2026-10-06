# Data Provenance & Schema Specification

This document details the structure, generation provenance, and format of the benchmark datasets supporting:
> **"Do Vision-Language Models Actually See?"**  
> Sachin Mishra & Meenakshi D’Souza (*IIIT Bangalore*)  
> Published in **Transactions on Machine Learning Research (09/2026)**  
> **OpenReview**: [https://openreview.net/forum?id=o9KC3v0pvb](https://openreview.net/forum?id=o9KC3v0pvb)

---

## 1. Stimulus Design Overview

The benchmark evaluates **15 quadruped mammal concepts** with strong, canonical four-legged priors:
* **Apex / High-Frequency**: Elephant, Lion, Horse, Deer, Dog, Cat.
* **Medium-Frequency**: Cheetah, Leopard, Fox, Hyena.
* **Low-Frequency / Tail**: Serval, Binturong, Okapi, Fossa, Numbat.

Each concept is tested across **4 factorial conditions**:
$$\{3\text{-legs}, 5\text{-legs}\} \times \{\text{Plain Background}, \text{Natural Context}\} = 4\text{ queries per concept}$$
$$\text{Total Stimuli} = 15 \times 4 = 60\text{ counterfactual queries}$$

---

## 2. Directory Structure

```
data/
├── image_manifest.csv                # Complete 60-image metadata and provenance manifest
├── processed/
│   ├── phase2_vlm_results.csv        # Per-query inference outputs for Llama-4-Scout (17B)
│   ├── llava_results.csv             # Per-query inference outputs for LLaVA-1.5 (7B Local)
│   ├── phase3_ollama_results.csv     # Local quantized model evaluation logs
│   └── model_confidence.csv          # Prior strength and lexical baseline data
└── raw/
    ├── counterfactual_images/        # 60 stimulus images (or via scripts/fetch_images.py)
    └── results_hf/                   # Raw inference logs for API / Hugging Face models
```

---

## 3. Scene Graph Evaluation Format (v2 Non-Leaking)

The Symbolic Bypass experiment provides models with structured JSON scene graphs isolating visual relational perception from semantics. To strictly prevent lexical leakage, all entity names, labels, and text descriptions are masked:

```json
{
  "image_id": "cheetah_5legs_natural",
  "entities": [
    {"id": "entity_0", "type": "subject"},
    {"id": "part_1", "type": "appendage"},
    {"id": "part_2", "type": "appendage"},
    {"id": "part_3", "type": "appendage"},
    {"id": "part_4", "type": "appendage"},
    {"id": "part_5", "type": "appendage"}
  ],
  "relations": [
    {"subject": "entity_0", "predicate": "has_part", "object": "part_1"},
    {"subject": "entity_0", "predicate": "has_part", "object": "part_2"},
    {"subject": "entity_0", "predicate": "has_part", "object": "part_3"},
    {"subject": "entity_0", "predicate": "has_part", "object": "part_4"},
    {"subject": "entity_0", "predicate": "has_part", "object": "part_5"}
  ]
}
```

* **Query Prompt**: `"Based strictly on the provided scene graph, how many appendages does the subject entity possess? Answer with a single integer."`

---

## 4. Hugging Face Hosting & Attribution

* **5-Legged Stimuli**: Sourced from Vo et al. (*"VLMs are Biased"*, arXiv:2410.02755).
* **3-Legged Stimuli**: Novel counterfactual inpainting suite generated using stable diffusion inpainting with manual limb erasure and artifact cleanup.
* **License**: Creative Commons Attribution 4.0 International ([CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)).
