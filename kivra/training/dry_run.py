"""Smoke test / dry-run mode: exercises the whole pipeline without model weights."""

from __future__ import annotations

import os

from ..config import PROFILES
from ..data.demo import write_demo_jsonl
from .sft import train


def run_demo_smoke_test(
    data_path: str = "data/demo.jsonl", out_dir: str = ".kivra-smoke-test"
) -> dict:
    """Generate demo data, ingest+splits, and run both profiles in dry-run mode."""
    write_demo_jsonl(data_path)
    results = {}
    for name in ("spark", "voltage"):
        profile = PROFILES[name]
        profile.train.output_dir = os.path.join(out_dir, name)
        info = train(profile, data_path=data_path, dry_run=True)
        results[name] = info
    return results


def main() -> None:
    results = run_demo_smoke_test()
    print("Kivra smoke test (dry-run, no weights):")
    for name, info in results.items():
        print(
            f"  {name}: {info['status']} — "
            f"train={info['train_examples']}, val={info['val_examples']}, "
            f"test={info['test_examples']}, base={info['base_model']}"
        )


if __name__ == "__main__":
    main()
