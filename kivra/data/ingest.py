"""Data ingestion pipeline.

Reads JSONL chat records, validates them against the schema, deduplicates exact
and near duplicates, produces a deterministic seeded train/val/test split, and
runs an eval-contamination check so held-out data never leaks from training.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

from .schema import Example, ValidationError, validate_example


@dataclass
class IngestReport:
    total_records: int
    valid_records: int
    invalid_records: int
    duplicates_removed: int
    train: int
    val: int
    test: int
    validation_errors: list[ValidationError]
    contamination: ContaminationReport


@dataclass
class ContaminationReport:
    train_eval_exact_overlap: int
    train_eval_high_overlap: int
    high_overlap_threshold: float
    flagged_eval_ids: list[str]


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _record_hash(example: Example) -> str:
    payload = json.dumps(
        [
            {"role": t.get("role"), "content": _normalize(t.get("content", ""))}
            for t in example.messages
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ngram_overlap(a: str, b: str, n: int = 5) -> float:
    ta = {_normalize(a)[i : i + n] for i in range(max(0, len(_normalize(a)) - n + 1))}
    tb = {_normalize(b)[i : i + n] for i in range(max(0, len(_normalize(b)) - n + 1))}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _text_of(example: Example) -> str:
    return "\n".join(f"{t.get('role')}: {t.get('content', '')}" for t in example.messages)


def load_jsonl(path: str) -> list[tuple[int, dict]]:
    """Read raw JSONL lines; returns (line_number, parsed_or_None) pairs."""
    records: list[tuple[int, dict]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((line_no, json.loads(line)))
            except json.JSONDecodeError:
                records.append((line_no, None))
    return records


def ingest(
    path: str,
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
    dedupe: bool = True,
    contamination_threshold: float = 0.75,
) -> tuple[list[Example], list[Example], list[Example], IngestReport]:
    """Ingest a JSONL file into validated, split examples.

    Returns (train_examples, val_examples, test_examples, report).
    """
    raw_records = load_jsonl(path)

    examples: list[Example] = []
    errors: list[ValidationError] = []
    for line_no, raw in raw_records:
        if raw is None:
            errors.append(
                ValidationError(line=line_no, record_id=f"line-{line_no}", errors=["invalid JSON"])
            )
            continue
        example, field_errors = validate_example(raw, line_no)
        errors.extend(field_errors)
        if not field_errors:
            examples.append(example)

    # Deduplication: exact hash of normalized message content.
    seen_hashes: set[str] = set()
    unique: list[Example] = []
    duplicates_removed = 0
    for ex in examples:
        h = _record_hash(ex)
        if dedupe and h in seen_hashes:
            duplicates_removed += 1
            continue
        seen_hashes.add(h)
        unique.append(ex)

    # Deterministic split: hash id + seed so the split is reproducible and
    # independent of file ordering.
    rng_key = f"{seed}:"

    def split_bucket(ex: Example) -> str:
        h = hashlib.sha256((rng_key + ex.id).encode("utf-8")).hexdigest()
        r = int(h, 16) % 10000 / 10000.0
        if r < train:
            return "train"
        if r < train + val:
            return "val"
        return "test"

    train_ex, val_ex, test_ex = [], [], []
    for ex in unique:
        bucket = split_bucket(ex)
        if bucket == "train":
            train_ex.append(ex)
        elif bucket == "val":
            val_ex.append(ex)
        else:
            test_ex.append(ex)

    # Contamination check on train vs eval (val + test).
    contamination = check_contamination(
        train_ex, val_ex + test_ex, threshold=contamination_threshold
    )

    report = IngestReport(
        total_records=len(raw_records),
        valid_records=len(unique),
        invalid_records=len(errors),
        duplicates_removed=duplicates_removed,
        train=len(train_ex),
        val=len(val_ex),
        test=len(test_ex),
        validation_errors=errors,
        contamination=contamination,
    )
    return train_ex, val_ex, test_ex, report


def check_contamination(
    train_examples: list[Example], eval_examples: list[Example], *, threshold: float = 0.75
) -> ContaminationReport:
    """Flag eval records that exactly duplicate or highly overlap train records."""
    train_texts = [_text_of(e) for e in train_examples]
    flagged: list[str] = []
    exact = 0
    high = 0
    for ex in eval_examples:
        text = _text_of(ex)
        is_exact = any(text == t for t in train_texts)
        best_overlap = max((_ngram_overlap(text, t) for t in train_texts), default=0.0)
        if is_exact:
            exact += 1
            flagged.append(ex.id)
        elif best_overlap >= threshold:
            high += 1
            flagged.append(ex.id)
    return ContaminationReport(
        train_eval_exact_overlap=exact,
        train_eval_high_overlap=high,
        high_overlap_threshold=threshold,
        flagged_eval_ids=flagged,
    )


def write_splits(
    out_dir: str,
    train_examples: list[Example],
    val_examples: list[Example],
    test_examples: list[Example],
) -> dict[str, str]:
    """Write splits as JSONL files; returns the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}
    for name, examples in (
        ("train", train_examples),
        ("val", val_examples),
        ("test", test_examples),
    ):
        p = os.path.join(out_dir, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n" for ex in examples)
        paths[name] = p
    return paths


def write_report(out_dir: str, report: IngestReport) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "ingest_report.json")
    payload = {
        "total_records": report.total_records,
        "valid_records": report.valid_records,
        "invalid_records": report.invalid_records,
        "duplicates_removed": report.duplicates_removed,
        "train": report.train,
        "val": report.val,
        "test": report.test,
        "validation_errors": [
            {"line": e.line, "record_id": e.record_id, "errors": e.errors}
            for e in report.validation_errors
        ],
        "contamination": asdict(report.contamination),
    }
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return p
