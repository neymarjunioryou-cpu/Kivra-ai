# Kivra

**Small footprint. Serious intelligence.**

Kivra is an open-weight language-model family built by fine-tuning suitably licensed Qwen checkpoints. It targets two release profiles:

| Profile | Target | Approx. size | Quantization | Suggested base |
|---|---|---|---|---|
| **Kivra Spark** | Low-end Android phones | 0.5B–1B | Int4 GGUF | Qwen3-0.6B Base |
| **Kivra Voltage** | PC / consumer GPU | ≥2B | Int4/Int8 GGUF | Qwen2.5-3B Base (or Qwen3-4B) |

**Proposed IDs (unverified for hub availability):** `kivra-spark-0.6b-instruct-v1`, `kivra-voltage-3b-instruct-v1`. Released names always use the *actual* base parameter count — fine-tuning does not change a checkpoint's size.

## Status: scaffold — no weights trained yet

This repository is a **runnable scaffold**, not a trained model. It contains:

- Configurable Spark and Voltage training profiles (LoRA / QLoRA)
- JSONL data ingestion with schema validation, deterministic splits, deduplication, and eval-contamination checks
- A tiny **synthetic demonstration dataset** (clearly marked; not production training data)
- A LoRA/QLoRA supervised fine-tuning trainer over Transformers + PEFT
- Fast smoke test / dry-run mode (no model weights downloaded)
- Evaluation scripts comparing base vs. adapted model (measured metrics only — no invented scores)
- Adapter merge/export and GGUF quantization workflow (llama.cpp)
- Local CLI inference and a minimal local chat interface (no API keys, no hosted dependency)
- Android deployment guide
- Model-card template, tests, lint config, sample configs

**Nothing is downloaded, trained, or published unless you run it.** No training or evaluation results are claimed in this repo.

## Quick start

```bash
# 1. Create a venv
python3 -m venv .venv && source .venv/bin/activate

# 2. Install core dependencies (data tools, smoke tests — no torch yet)
pip install -e ".[dev]"

# 3. Run the smoke test (dry-run, no weights, ~seconds)
python scripts/train.py --profile spark --dry-run
python scripts/train.py --profile voltage --dry-run

# 4. Inspect the synthetic demo data
head -n 3 data/demo.jsonl
```

To train real weights you additionally need `torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes` (install with `pip install -e ".[train]"`), a Hugging Face hub account with access to the base checkpoint, and a GPU for anything beyond toy experiments. See [TRAINING](#training-real-weights) below.

## Repository layout

```
kivra/          # package: config, data, training, eval, inference, storage
configs/        # spark.yaml, voltage.yaml training profiles
data/           # demo.jsonl (tiny synthetic demo dataset only)
docs/           # BASE_SELECTION.md, GGUF_EXPORT.md, STORAGE.md
modelcard/      # TEMPLATE.md
scripts/        # train.py, evaluate.py, infer.py, export_gguf.sh
tests/          # pytest suite
android/        # DEPLOYMENT.md
```

## Configuration

Sample profiles live in `configs/spark.yaml` and `configs/voltage.yaml`. Edit copies, not the samples.

```bash
python scripts/train.py --profile spark   --config my_spark.yaml   --data my_data.jsonl
python scripts/train.py --profile voltage --config my_voltage.yaml --data my_data.jsonl
```

All settings (base model, LoRA rank, batch size, sequence length, seed, output dir) are configurable in YAML or via `kivra.config`.

## Data ingestion

Place your training data as JSONL, one object per line:

```json
{"id": "demo-001", "messages": [{"role": "system", "content": "You are Kivra, a helpful assistant."}, {"role": "user", "content": "Say hello."}, {"role": "assistant", "content": "Hello!"}], "source": "my-corpus", "task": "chat", "lang": "en"}
```

Run ingestion:

```bash
python -m kivra.data.ingest data/my_data.jsonl --out data/processed --seed 42 --train 0.8 --val 0.1 --test 0.1
```

The pipeline validates each record against the schema, deduplicates exact/near duplicates, makes a **deterministic, seeded split**, and runs an **eval-contamination check** that flags train/eval overlap (exact matches and high n-gram overlap) so you do not evaluate on training data. See `kivra/data/ingest.py` and `docs/` for details.

## Training real weights

1. Pick a profile: `spark` (Qwen3-0.6B, QLoRA 4-bit, phone-friendly) or `voltage` (Qwen2.5-3B or Qwen3-4B, LoRA, PC).
2. Review `docs/BASE_SELECTION.md` for the base-selection rationale, license (Apache 2.0), and attribution duties.
3. Prepare JSONL data and run ingestion.
4. Launch training:

```bash
pip install -e ".[train]"
python scripts/train.py --profile spark --data data/processed/train.jsonl
```

Training saves **LoRA adapters only** by default (small, redistributable without re-publishing the base weights). Resume from checkpoints automatically; all runs are seeded for reproducibility.

Hardware rule of thumb:
- Spark: 4-bit QLoRA on Qwen3-0.6B fits on a single T4/16 GB GPU (or CPU, slowly). Fine for a fine-tune demo.
- Voltage: Qwen2.5-3B/4B LoRA wants a GPU with ≥12–24 GB VRAM depending on quantization and sequence length; 4-bit QLoRA reduces this substantially.

**Do not launch paid training jobs without approving the cost first.** This repo launches nothing on its own.

## Evaluation

```bash
python scripts/evaluate.py --base Qwen/Qwen3-0.6B-Base --adapter runs/kivra-spark --data data/processed/test.jsonl
```

Reports measured perplexity, generation latency (tokens/sec), and memory usage on the held-out set only. Scores are **not** invented and are not benchmark-quality unless you say so.

## Export & quantization

Merge adapters into the base and export to GGUF for llama.cpp:

```bash
python -m kivra.export.merge_gguf --adapter runs/kivra-spark --base Qwen/Qwen3-0.6B-Base --out kivra-spark-merged
./scripts/export_gguf.sh kivra-spark-merged kivra-spark-merged.Q4_K_M.gguf
```

See `docs/GGUF_EXPORT.md` for the llama.cpp workflow and the difference between adapter-only and merged-weight distribution.

## Local inference (no API keys)

```bash
python scripts/infer.py --base Qwen/Qwen3-0.6B-Base --adapter runs/kivra-spark --prompt "Hello, Kivra!"
python -m kivra.inference.chat          # minimal local chat loop
```

Everything runs locally with Hugging Face Transformers; no hosted API, no API key required for normal local inference.

## Android deployment

See `android/DEPLOYMENT.md`: llama.cpp on Android is the recommended runtime; Qwen3-0.6B quantized to Q4_K_M is roughly 1 GB on disk and ~1–1.5 GB in RAM with a small context. Collect device benchmarks: time-to-first-token, tokens/sec, peak RAM, APK size.

## Cloud / external storage (optional, off by default)

Workspaces and laptops run out of space. `kivra/storage/connectors.py` plus `scripts/storage.py` let you offload datasets and checkpoints to **Supabase Storage, Dropbox, or Google Drive** — free tiers shown below.

Nothing is uploaded automatically. Configure exactly one backend in `.env` (copy `.env.example`), then use the CLI:

```bash
# 1. Install optional SDKs
pip install -e ".[storage]"

# 2. Configure .env (one backend only)
cp .env.example .env   # edit SUPABASE_URL / SUPABASE_SERVICE_KEY / bucket, etc.

# 3. Load credentials and offload a checkpoint directory
set -a; . .env; set +a
python scripts/storage.py --backend supabase --bucket kivra-checkpoints upload runs/kivra-spark remote/kivra-spark

# 4. List / download
python scripts/storage.py --backend supabase --bucket kivra-checkpoints list remote/kivra-spark
python scripts/storage.py --backend supabase --bucket kivra-checkpoints download remote/kivra-spark/adapter_model.safetensors restored.safetensors
```

| Backend | Free tier | Best for |
|---|---|---|
| Supabase Storage | 1 GB | LoRA adapters, small GGUF exports |
| Dropbox | 2 GB | Datasets + adapters |
| Google Drive | 15 GB (shared) | Datasets + adapters + small GGUF |

Full base checkpoints (several GB each) are best re-downloaded from Hugging Face rather than mirrored, to respect redistribution terms and keep provenance clear.

See `docs/STORAGE.md` and `scripts/storage.py --help`.

## Model cards

Fill in `modelcard/TEMPLATE.md` before releasing anything: base checkpoint, license, dataset provenance, training settings, limitations, and (only measured) evaluation results.

## Testing & lint

```bash
pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

## .gitignore

Datasets, checkpoints, adapters, GGUF files, `.env` secrets, and caches are excluded by default. Do not commit weights or private data.

## License

MIT (see `LICENSE`). Base Qwen checkpoints remain under their original **Apache 2.0** license; attribution and redistribution requirements of the base checkpoint apply to any released Kivra weights.

## Contributing

See `CONTRIBUTING.md`. Ask before spending money, publishing weights, or pushing large artifacts.
