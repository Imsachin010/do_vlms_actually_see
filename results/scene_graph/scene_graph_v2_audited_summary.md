# Audited Scene Graph v2 Benchmark Summary (Leakage-Verified 60 JSONs)

| Model | Regime | Backend | Valid Queries | Correct | Scene Graph Acc. (%) | Raw Image Acc. (%) | Gain (pp) | Perception Error Rate (%) |
|---|---|---|---|---|---|---|---|---|
| **Llama-4-Scout** | Regime A | Groq API | 60/60 | 60 | **100.0%** | 13.3% | **+86.7pp** | **86.7%** |
| **Gemma-4-31B** | Regime A | HF API | 60/60 | 60 | **100.0%** | 11.9% | **+88.1pp** | **88.1%** |
| **Qwen3-VL-235B** | Regime A | Novita API | 60/60 | 60 | **100.0%** | 0.0% | **+100.0pp** | **100.0%** |
| **Qwen2.5-7B (Ollama)** | Regime B | Ollama (Q4_0 Local) | 60/60 | 50 | **83.3%** | 0.0% | **+83.3pp** | N/A (Regime B) |
| **LLaVA-1.5 (Ollama)** | Regime B | Ollama (Q4_0 Local) | 60/60 | 39 | **65.0%** | 0.0% | **+65.0pp** | N/A (Regime B) |
| **MiniCPM-V (Ollama)** | Regime B | Ollama (Q4_0 Local) | 60/60 | 30 | **50.0%** | 0.0% | **+50.0pp** | N/A (Regime B) |
| **Gemma-3-4B (HF)** | Regime B | HF API | 60/60 | 6 | **10.0%** | 0.0% | **+10.0pp** | N/A (Regime B) |
| **Llama-3.2-Vision (Ollama)** | Regime B | Ollama (Q4_0 Local) | 60/60 | 0 | **0.0%** | 0.0% | **+0.0pp** | N/A (Regime B) |