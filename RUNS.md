# Model Evaluation Runs & Environment Inventory

This document inventories the 14 vision-language model configurations evaluated across the two experimental regimes.

---

## 1. Experimental Settings

All queries were executed deterministically:
* **Temperature**: `0.0`
* **Seed**: `42` (where supported by API/engine)
* **Sampling**: Greedy decoding (`do_sample=False`)
* **Maximum Tokens**: `20` (integer-focused answer)

---

## 2. Model Configuration Inventory

| Model Family | Configuration | Parameters | Engine / Backend | Precision | Result CSV Path | Regime |
|---|---|---|---|---|---|---|
| **Llama-4** | Llama-4-Scout | 17B | Groq API | FP16 | `data/processed/phase2_vlm_results.csv` | **Regime A (Pass)** |
| **Gemma-4** | Gemma-4-31B | 31B | Hugging Face Router | FP16 | `data/raw/results_hf/gemma_4_31b_results.csv` | **Regime A (Pass)** |
| **Qwen3** | Qwen3-VL-235B-A22B | 235B | Novita API | FP16 | `data/raw/results_hf/qwen3_vl_235b_a22b_results.csv` | **Regime A (Pass)** |
| **LLaVA-v1.6** | LLaVA-v1.6-Mistral | 7B | PyTorch / Transformers | FP16 | `results/scene_graph_v2_inference_required/llava_v16_mistral.csv` | **Regime A (Pass)** |
| **InstructBLIP** | InstructBLIP-FlanT5-XL | 4B | PyTorch / Transformers | FP16 | `results/scene_graph_v2_inference_required/instructblip_t5xl.csv` | **Regime A (Pass)** |
| **Qwen2.5** | Qwen2.5-VL | 7B | Hugging Face Router | FP16 | `results/scene_graph_v2_inference_required/qwen25_vl_fp16_60queries.csv` | **Regime B (Borderline 90.0%)** |
| **Qwen2.5** | Qwen2.5-VL | 7B | Ollama | Q4_0 | `results/scene_graph_v2_inference_required/qwen25_7b_ollama.csv` | **Regime B (Fail 83.3%)** |
| **LLaVA-1.5** | LLaVA-1.5 | 7B | Ollama | Q4_0 | `results/scene_graph_v2_inference_required/llava_15_ollama.csv` | **Regime B (Fail 65.0%)** |
| **LLaVA-1.5** | LLaVA-1.5 | 7B | PyTorch / Transformers | FP16 | `results/scene_graph_v2_inference_required/llava_15_hf_fp16.csv` | **Regime B (Fail 50.0%)** |
| **MiniCPM-V** | MiniCPM-V 2.6 | 8B | Ollama | Q4_0 | `results/scene_graph_v2_inference_required/minicpm_v.csv` | **Regime B (Fail 50.0%)** |
| **Qwen2-VL** | Qwen2-VL | 2B | PyTorch / Transformers | FP16 | `results/scene_graph_v2_inference_required/qwen2_vl.csv` | **Regime B (Fail 50.0%)** |
| **SmolVLM2** | SmolVLM2 | 2.2B | PyTorch / Transformers | FP16 | `results/scene_graph_v2_inference_required/smolvlm2.csv` | **Regime B (Fail 46.7%)** |
| **Gemma-3** | Gemma-3-4B | 4B | PyTorch / Transformers | FP16 | `results/scene_graph_v2_inference_required/gemma_3_4b.csv` | **Regime B (Fail 3.3%)** |
| **Llama-3.2** | Llama-3.2-Vision | 11B | Ollama | Q4_0 | `results/scene_graph_v2_inference_required/llama32_vision.csv` | **Regime B (Fail 0.0%)** |

---

## 3. Hardware & Runtime Environment

* **Local Inference**: NVIDIA GeForce RTX GPU (16GB VRAM), CUDA 12.8, PyTorch 2.9.1, Transformers 4.57.5.
* **Quantized Runs**: Ollama v0.5+ running 4-bit integer quantized GGUF weights.
* **Cloud API Inference**: Groq, Novita AI, Hugging Face Serverless Inference.
