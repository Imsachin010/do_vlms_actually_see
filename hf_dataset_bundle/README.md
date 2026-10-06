---
language:
- en
license: cc-by-4.0
tags:
- vision-language-models
- bias-evaluation
- counterfactuals
- counting
- scene-graphs
task_categories:
- visual-question-answering
- object-detection
pretty_name: "Counterfactual Quadruped Visual Bias Dataset"
size_categories:
- n<1K
---

# Counterfactual Quadruped Visual Bias Dataset (N=60)

## Dataset Summary
This dataset contains **60 counterfactual quadruped images** designed to evaluate visual feature counting versus semantic prior collapse in Vision-Language Models (VLMs).

Supporting the paper:
> **"Do Vision-Language Models Actually See?"**  
> Sachin Mishra & Meenakshi D’Souza (*IIIT Bangalore*)  
> Published in **Transactions on Machine Learning Research (09/2026)**  
> **OpenReview**: [https://openreview.net/forum?id=o9KC3v0pvb](https://openreview.net/forum?id=o9KC3v0pvb)

The dataset covers **15 animal concepts** with strong prior canonical leg counts of 4 (Elephant, Lion, Deer, Horse, Cheetah, Leopard, Fox, Hyena, Cat, Dog, Serval, Binturong, Okapi, Fossa, Numbat):
* **2 Counterfactual Leg Counts**: 3 legs and 5 legs.
* **2 Background Contexts**: Plain (isolated/white) and Natural Context.
* **Total**: $15 \times 2 \times 2 = 60$ counterfactual stimulus images.

## Directory Structure
```
├── README.md               # Dataset Card with schema and provenance
├── image_manifest.csv      # Complete metadata and attribution manifest
└── images/                 # 60 stimulus images (.png / .jpg)
```

## Provenance & Attribution
* **5-Legged Stimuli**: Derived from the benchmark suite by Vo et al. (*"VLMs are Biased"*, arXiv:2410.02755).
* **3-Legged Stimuli**: Synthesized counterfactual limb-ablation edits created specifically to test symmetry breaking and prior pull directionality ($3 \rightarrow 4$ vs. $5 \rightarrow 4$).
* **License**: Creative Commons Attribution 4.0 International ([CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)).

## Usage in Python
```python
import pandas as pd
from PIL import Image

manifest = pd.read_csv("image_manifest.csv")
for _, row in manifest.iterrows():
    img = Image.open(f"images/{row['image_filename']}")
    print(f"Loaded {row['concept']} with {row['counterfactual_legs']} legs ({row['background']})")
```

## Citation
```bibtex
@article{mishra2026dovisionlanguagemodels,
  title={Do Vision-Language Models Actually See?},
  author={Mishra, Sachin and D'Souza, Meenakshi},
  journal={Transactions on Machine Learning Research},
  year={2026},
  month={September},
  url={https://openreview.net/forum?id=o9KC3v0pvb}
}
```
