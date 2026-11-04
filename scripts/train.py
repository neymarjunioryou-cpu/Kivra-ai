"""Entry point: train a Kivra profile (Spark or Voltage) with LoRA / QLoRA."""

from __future__ import annotations

import argparse
import sys

from kivra.config import load_profile
from kivra.training.sft import load_profile_entry, train


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train a Kivra model profile")
    parser.add_argument("--profile", choices=["spark", "voltage"], help="Built-in profile name")
    parser.add_argument(
        "--config", default=None, help="Path to a YAML config (overrides --profile)"
    )
    parser.add_argument("--data", required=True, help="Training JSONL file")
    parser.add_argument(
        "--output-dir", default=None, help="Output directory for adapters/checkpoints"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Smoke test: validate config and data, no weights"
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args(argv)

    if args.config:
        profile = load_profile(path=args.config)
    else:
        profile = load_profile_entry(args.profile or "spark")

    if args.output_dir:
        profile.train.output_dir = args.output_dir

    try:
        info = train(
            profile,
            data_path=args.data,
            dry_run=args.dry_run,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    except Exception as exc:  # pragma: no cover - surface clearly to user  # noqa: BLE001
        print(f"Kivra training failed: {exc}", file=sys.stderr)
        return 1

    print("Kivra train result:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
