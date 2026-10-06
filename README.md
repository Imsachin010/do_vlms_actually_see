# Do Vision-Language Models Actually See?

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests: Pytest](https://img.shields.io/badge/pytest-6%20passed-brightgreen.svg)](tests/test_paper_numbers.py)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Reproducibility Certification](https://img.shields.io/badge/Reproducibility-Certified-success.svg)](REPRODUCIBILITY.md)
[![OpenReview](https://img.shields.io/badge/OpenReview-o9KC3v0pvb-red.svg)](https://openreview.net/forum?id=o9KC3v0pvb)

This repository contains the official evaluation codebase, audited per-query inference logs, and analysis suite for the paper:
> **"Do Vision-Language Models Actually See?"**  
> **Sachin Mishra** ([sachin.mishra@iiitb.ac.in](mailto:sachin.mishra@iiitb.ac.in)) — *Department of Artificial Intelligence and Data Science, IIIT Bangalore*  
> **Meenakshi D’Souza** ([meenakshi@iiitb.ac.in](mailto:meenakshi@iiitb.ac.in)) — *Department of Computer Science and Engineering, IIIT Bangalore*  
> **Transactions on Machine Learning Research (TMLR), September 2026**  
> **OpenReview**: [https://openreview.net/forum?id=o9KC3v0pvb](https://openreview.net/forum?id=o9KC3v0pvb)

---

## 🚀 Quickstart (Reproduce Paper Numbers in < 3 Seconds)

```bash
# 1. Clone repository
git clone https://github.com/<username>/vlm-bias-prediction.git
cd vlm-bias-prediction

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Verify all published numbers against raw audited logs
pytest tests/test_paper_numbers.py -v
```

---

## 📊 Reproduction & Audit Pipeline

Complete reproduction commands are mapped in **[REPRODUCIBILITY.md](file:///c:/Users/Imsac/Desktop/Project_X/X5/vlm_bias_prediction/REPRODUCIBILITY.md)**:

### 1. Print Paper Tables & Hypothesis Statistics
```bash
# Print Tables 2 & 4: Perception Bypass Ablation across all 14 model configurations
python src/analyze_audit.py --table gate

# Print Table 7 & Appendix E.4: Hypothesis 2 Visual Salience Spearman Correlation & OLS
python src/analyze_audit.py --h2

# Print Table 14: Family-Wise Error Rate (FWER) Holm-Bonferroni Corrections
python src/analyze_audit.py --holm

# Print all tables and audits
python src/analyze_audit.py --all
```

### 2. Regenerate All Publication Figures
```bash
# Regenerates Figures 1–5 deterministically into assets/
python src/make_figures.py --all
```

---

## 🗂️ Repository Structure

```
├── REPRODUCIBILITY.md          # Step-by-step reproduction guide and table mapping
├── DATA.md                     # Dataset documentation, provenance, and annotation schema
├── RUNS.md                     # Complete inventory of 14 model configurations and backends
├── LICENSE                     # MIT open-source license
├── CITATION.cff                # Citation metadata
├── requirements.txt            # Strictly pinned dependency versions
│
├── analyze_audit.py            # Unified paper table and statistical test runner
├── make_figures.py             # Single-entry script to regenerate Figures 1–5
│
├── tests/
│   └── test_paper_numbers.py   # Automated pytest assertion suite for all published metrics
│
├── scripts/
│   └── fetch_images.py         # Downloader script for Hugging Face image stimuli
│
├── data/
│   ├── image_manifest.csv      # Complete metadata and attribution manifest
│   └── processed/              # Audited per-query inference result CSVs
│
├── scene_graph/
│   ├── data/                   # 60 leakage-verified Scene Graph v2 JSONs
│   └── results/                # Scene graph symbolic evaluation logs
│
└── assets/                     # Deterministically generated vector PDFs & preview PNGs
```

---

## 📦 Dataset Access & Stimulus Images

To keep Git repository clones lightweight, the 60 counterfactual stimulus images are hosted on Hugging Face:

```bash
python scripts/fetch_images.py --dest data/counterfactual_images
```

* **Dataset Card & Provenance**: See **[DATA.md](file:///c:/Users/Imsac/Desktop/Project_X/X5/vlm_bias_prediction/DATA.md)** and `hf_dataset_bundle/README.md`.
* **5-Legged Stimuli**: Derived from Vo et al. (*"VLMs are Biased"*, arXiv:2410.02755).
* **3-Legged Stimuli**: Novel counterfactual inpainting suite created for this study (CC-BY-4.0).

---

## ⚖️ License & Citation

* **Code**: [MIT License](LICENSE)
* **Dataset & Annotations**: [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)

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
