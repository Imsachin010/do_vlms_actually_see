# Reproducibility Guide

This document provides exact, deterministic commands to reproduce all tables, figures, statistical tests, and audits reported in the paper:
> **"Do Vision-Language Models Actually See?"**  
> Sachin Mishra & Meenakshi D’Souza (*IIIT Bangalore*)  
> Published in **Transactions on Machine Learning Research (09/2026)**  
> **OpenReview**: [https://openreview.net/forum?id=o9KC3v0pvb](https://openreview.net/forum?id=o9KC3v0pvb)

---

## 1. Quickstart & Verification

Verify all published results against the paper's claimed numerical values in **< 3 seconds**:

```bash
# Install pinned dependencies
pip install -r requirements.txt

# Run full automated test suite asserting paper numbers
pytest tests/test_paper_numbers.py -v
```

Expected output:
```
tests/test_paper_numbers.py::test_scout_raw_metrics PASSED               [ 16%]
tests/test_paper_numbers.py::test_llava_raw_metrics PASSED               [ 33%]
tests/test_paper_numbers.py::test_regime_a_gate_passers PASSED           [ 50%]
tests/test_paper_numbers.py::test_prior_in_text_zero PASSED              [ 66%]
tests/test_paper_numbers.py::test_paired_background_statistics PASSED    [ 83%]
tests/test_paper_numbers.py::test_visual_salience_spearman PASSED        [100%]
============================== 6 passed in 2.44s ==============================
```

---

## 2. Table-to-Code Mapping

| Paper Table | Description | Execution Command | Output / Metrics |
|---|---|---|---|
| **Tables 2 & 4** | Raw Image vs. Scene Graph Accuracy Ablation across all 14 configurations | `python src/analyze_audit.py --table gate` | Gate verification: 5 Regime A passers (100%), Regime B breakdown, 0.0% prior-in-text pull. |
| **Table 7 & App. E.4** | Hypothesis 2: Visual Salience vs. Bias Rate & Multiple OLS Regression | `python src/analyze_audit.py --h2` | Spearman $\rho = -0.604$ ($p = 0.017$), Theil-Sen fit, frequency vs. visual separability OLS. |
| **Table 10** | Paired Background Context Bias Effect (Plain vs. Natural) | `python verify_raw_background_invariants.py` | LLaVA-1.5 ($t(14)=1.00, p=0.334$), Scout ($t(14)=0.323, p=0.751$). |
| **Table 14** | Holm-Bonferroni Family-Wise Error Rate (FWER) Adjustments | `python src/analyze_audit.py --holm` | Raw $p$-values vs. Holm-adjusted significance across all 6 hypothesis families. |
| **Export All** | Export verified summary CSVs into `analysis/` | `python src/analyze_audit.py --export` | Backs Table 4 and Table 5 summaries. |
| **All Tables** | Unified execution of all tables and audits | `python src/analyze_audit.py --all` | Complete printout of paper statistics. |

---

## 3. Figure Generation

All publication figures (PDF and high-resolution PNG) are deterministically compiled into `assets/`:

```bash
# Compile all figures (Figures 1-5)
python src/make_figures.py --all

# Or compile individual figures
python src/make_figures.py --fig 1   # Figure 1: Overall Bias Rates
python src/make_figures.py --fig 2   # Figure 2: Visual Salience vs. Bias Rate
python src/make_figures.py --fig 3   # Figure 3: Perception Bypass Ablation
python src/make_figures.py --fig 4   # Figure 4: Stacked Error Distribution
python src/make_figures.py --fig 5   # Figure 5: Background Context Effect
```

* **Figure 1** (`assets/Figure_Overall_Bias_Horizontal.pdf`): Overall bias rate ranking across 12 vision-language architectures.
* **Figure 2** (`assets/figure_salience_scout.pdf`): Visual feature salience vs. bias rate for Llama-4-Scout with Theil-Sen regression and 95% bootstrap band.
* **Figure 3** (`assets/figure_perception_ablation.pdf`): Comprehensive 14-configuration perception bypass ablation comparing Image vs. Scene Graph accuracy with Wilson score 95% CIs.
* **Figure 4** (`assets/Figure_Error_Distribution_Stacked.pdf`): 4-category audited error distribution (prior-aligned, off-by-one, visual confusion, other).
* **Figure 5** (`assets/figure_background_effect.pdf`): Paired background context comparison (Plain vs. Natural Context).

---

## 4. Dataset & Stimulus Acquisition

Image stimuli are hosted on Hugging Face to avoid large binary commits in Git:
```bash
python scripts/fetch_images.py --dest data/counterfactual_images
```
* **5-Legged Stimuli**: Vo et al. (*"VLMs are Biased"*, arXiv:2410.02755).
* **3-Legged Stimuli**: Novel counterfactual inpainting suite created for this study (CC-BY-4.0).
* **Dataset Manifest**: [data/image_manifest.csv](file:///c:/Users/Imsac/Desktop/Project_X/X5/vlm_bias_prediction/hf_dataset_bundle/image_manifest.csv).
