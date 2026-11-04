# Base model selection — Kivra Spark & Voltage

## Requirements from the brief

- **Kivra Spark (phone):** ~0.5B–1B parameters, quantized for low-end Android.
- **Kivra Voltage (PC):** at least 2B parameters if feasible with the selected base and hardware.
- Released names must use the *actual* base parameter count — fine-tuning does not change checkpoint size.

## Candidate family: Qwen (Alibaba Cloud)

Qwen checkpoints (Qwen3 and Qwen2.5) are released under **Apache 2.0**, support
commercial use, have Hugging Face/ModelScope availability, GGUF/llama.cpp
support, and good Transformers/PEFT tooling. Attribution: keep the Qwen model
card/license notice with any derivative release.

| Base checkpoint | Params | License | Context | Notes |
|---|---|---|---|---|
| Qwen/Qwen3-0.6B-Base | ~0.6B | Apache 2.0 | 32K | Fits Spark 0.5B–1B target; smallest Qwen3 dense |
| Qwen/Qwen3-1.7B-Base | ~1.7B | Apache 2.0 | 32K | Above Spark range; strong PC option |
| Qwen/Qwen2.5-3B-Base | 3B | Apache 2.0 | 32K | Meets Voltage "≥2B" rule with actual count |
| Qwen/Qwen3-4B-Base | ~4B | Apache 2.0 | 32K | Meets Voltage rule; more VRAM needed |

## Recommendation

**Kivra Spark → Qwen3-0.6B Base** → proposed id `kivra-spark-0.6b-instruct-v1`.
- Within the 0.5B–1B phone target.
- Q4_K_M GGUF ≈ 1 GB on disk, ~1–1.5 GB RAM at small context — viable on low-end Android.
- Apache 2.0, 32K context, strong multilingual coverage (119 languages).

**Kivra Voltage → Qwen2.5-3B Base** → proposed id `kivra-voltage-3b-instruct-v1`.
- Meets the ≥2B requirement using the *actual* 3B count (per the brief: do not
  silently rename a 1.5B base to 2B, and do not claim fine-tuning grows parameters).
- Fits consumer GPUs with LoRA; 4-bit QLoRA keeps VRAM modest.
- If you have ≥16–24 GB VRAM and want more quality, substitute Qwen3-4B and name
  the release `kivra-voltage-4b-instruct-v1` — never `...-2b-...` unless the
  base is actually 2B.

## What we do not claim

- Frontier performance from a small fine-tune.
- That training or evaluation has been run (it has not — this is a scaffold).
- Hub name availability for the proposed IDs (unverified).

## Hardware budget note

Before launching any paid training job, approve the cost. QLoRA on Qwen3-0.6B
fits a single T4-class GPU; Qwen2.5-3B/4B LoRA wants ≥12–24 GB VRAM depending
on quantization and sequence length.
