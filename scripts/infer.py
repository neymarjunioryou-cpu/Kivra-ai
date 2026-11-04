"""Entry point: one-shot local inference."""

from __future__ import annotations

import argparse
import sys

from kivra.inference.cli import generate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Kivra local inference")
    parser.add_argument("--base", required=True, help="Base model repo id or path")
    parser.add_argument("--adapter", default=None, help="Adapter directory (optional)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args(argv)
    try:
        text = generate(
            args.base,
            adapter=args.adapter,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            load_in_4bit=args.load_in_4bit,
        )
        print(text)
    except Exception as exc:  # pragma: no cover  # noqa: BLE001
        print(f"Kivra inference failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
