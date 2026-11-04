# Model card — Kivra <spark|voltage>-<size>-instruct-v1

> Fill every section before releasing. Report only metrics you actually measured;
> never invent scores.

## 1. Model details
- **Model id:** kivra-spark-0.6b-instruct-v1 / kivra-voltage-3b-instruct-v1
- **Developer:** Kivra contributors
- **Release date:** (unset)
- **Version:** 0.1
- **Paper or source:** none (community fine-tune)

## 2. Base checkpoint
- **Base model:** Qwen/Qwen3-0.6B-Base (or Qwen/Qwen2.5-3B-Base)
- **License:** Apache 2.0 (Qwen) — include base model notice with any release
- **Attribution:** derived from Qwen by Alibaba Cloud; fine-tuned by Kivra contributors
- **Download:** https://huggingface.co/Qwen

## 3. Intended use
- Chat / assistant on low-end Android (Spark) or PC (Voltage).
- Local, offline inference; no API key required.

## 4. Training data
- **Provenance:** (describe source, rights, cleaning)
- **License of data:** (unset)
- **Size:** (unset) records, (unset) tokens
- **Dedup / splits:** deterministic seeded split, eval-contamination checked (see `kivra/data/ingest.py`)
- **Demo data:** `data/demo.jsonl` is synthetic and NOT used for production releases

## 5. Training settings
- **Method:** LoRA / QLoRA supervised fine-tuning (PEFT)
- **Framework:** Transformers + PEFT + Accelerate
- **Seed:** 42
- **Config:** `configs/spark.yaml` / `configs/voltage.yaml` (attach exact file used)

## 6. Evaluation
- **Tasks:** (unset)
- **Metrics:** measured perplexity / latency / memory only (see `scripts/evaluate.py`)
- **Results:** (unset) — not benchmark-quality unless stated

## 7. Limitations
- Small fine-tune; not frontier-quality.
- Quantization may degrade quality; verify after Q4/Q8 conversion.
- Tested minimally; no large-scale benchmarking performed.

## 8. Ethical considerations
- Qwen Apache 2.0 terms apply; review use restrictions of the base checkpoint.
- Do not deploy for high-stakes decisions without independent evaluation.

## 9. Citation
- Qwen team (Alibaba Cloud). Qwen models. https://github.com/QwenLM/Qwen
