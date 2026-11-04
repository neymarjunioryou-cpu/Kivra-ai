"""Entry point: compare base vs adapted model on held-out data."""

from __future__ import annotations

import argparse
import sys

from kivra.eval.compare import compare


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Kivra base vs adapted model")
    parser.add_argument("--base", required=True, help="Base model repo id or path")
    parser.add_argument("--adapter", default=None, help="Adapter directory (optional)")
    parser.add_argument("--data", default=None, help="Held-out data file")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--out", default=None, help="Write metrics JSON here")
    args = parser.parse_args(argv)
    try:
        compare(
            args.base,
            adapter=args.adapter,
            data_path=args.data,
            load_in_4bit=args.load_in_4bit,
            out_path=args.out,
        )
    except Exception as exc:  # pragma: no cover  # noqa: BLE001
        print(f"Kivra evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
