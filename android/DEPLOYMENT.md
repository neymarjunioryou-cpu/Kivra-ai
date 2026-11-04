# Android deployment guide — Kivra Spark

## Recommended runtime: llama.cpp on Android

Build llama.cpp for Android (NDK) and run the quantized GGUF (`Kivra Spark`:
Qwen3-0.6B Q4_K_M ≈ 1 GB on disk).

Alternative runtimes to evaluate: MLC LLM Android, MediaPipe LLM Inference,
onnxruntime-mobile. Pick one and benchmark before committing.

## Realistic memory / context guidance (Qwen3-0.6B, Q4_K_M)

| Item | Approximation |
|---|---|
| Model on disk | ~1 GB (Q4_K_M) |
| Model in RAM | ~1.1–1.3 GB |
| KV cache | ~0.5–1 MB per 1K tokens at 4-bit (context dependent) |
| Suggested context | 512–2048 tokens on low-end devices (2–4 GB RAM phones) |
| App size impact | +~1–2 GB (GGUF) — consider dynamic feature download |

Keep context short on 2 GB RAM devices; always measure on target hardware.

## Device benchmarks to collect

Before declaring a release, measure on representative devices:

1. Time to first token (TTFT) — ms
2. Generation speed — tokens/sec
3. Peak RAM — MB (process RSS + GPU if applicable)
4. App install size — MB
5. Battery temperature / thermal throttle over a 5-minute generation
6. First-token vs steady-state latency difference

Record device model, RAM, Android version, chipset, quant type, and context
length with every result.

## Steps

1. Train + export GGUF (see `docs/GGUF_EXPORT.md`).
2. Copy `.gguf` into `app/src/main/assets/` (or download at install time).
3. Initialize llama.cpp Android bindings, load model with small `n_ctx`.
4. Implement prompt → tokenize → generate → detokenize loop.
5. Benchmark, then tune `n_ctx` and batch/threads per device class.
