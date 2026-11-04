# Contributing to Kivra

Kivra is a scaffold for fine-tuning small Qwen-based language models. Keep it runnable, honest, and dependency-light.

## Ground rules

1. **No fabricated results.** Never claim training, evaluation, or benchmark numbers you did not actually run. Report measured metrics only, labeled as such.
2. **No secrets, no weights.** Datasets, checkpoints, adapters, GGUF files, and `.env` files are gitignored. Do not commit them.
3. **Ask before spending.** Do not launch paid training jobs, publish weights, or push large artifacts without approval.
4. **Preserve attribution.** Qwen checkpoints are Apache 2.0; keep the base model, license, and dataset provenance in every model card.

## Workflow

1. Fork/clone on branch `arena/6d259a05-kivra-ai` (this session's branch) or your own feature branch.
2. Make changes, run `pytest` and `ruff check .`.
3. Open a pull request describing what changed and whether you trained/evaluated anything.

## Conventions

- Python 3.9+, type hints with `from __future__ import annotations`.
- One config dataclass per concern (`kivra/config.py`).
- Cloud storage is optional and off by default (`kivra/storage/`).
