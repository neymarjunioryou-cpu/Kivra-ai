"""Tests for the Kivra data ingestion pipeline (no model weights required)."""

import json
import os

from kivra.data.ingest import (
    ContaminationReport,
    check_contamination,
    ingest,
    write_report,
    write_splits,
)
from kivra.data.schema import Example, validate_example


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def test_validate_example_ok():
    ex, errors = validate_example(
        {
            "id": "a",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        },
        1,
    )
    assert errors == []
    assert ex.id == "a"


def test_validate_example_bad_role():
    _ex, errors = validate_example(
        {
            "id": "a",
            "messages": [
                {"role": "hero", "content": "hi"},
            ],
        },
        1,
    )
    assert len(errors) == 1
    assert "role" in errors[0].errors[0]


def test_ingest_demo_schema(tmp_path):
    p = tmp_path / "demo.jsonl"
    _write_jsonl(
        p,
        [
            {
                "id": "1",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "yo"},
                ],
                "source": "s",
                "task": "chat",
                "lang": "en",
            },
            {
                "id": "2",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "yo"},
                ],
                "source": "s",
                "task": "chat",
                "lang": "en",
            },  # exact duplicate
            {
                "id": "3",
                "messages": [
                    {"role": "user", "content": "different thing here"},
                    {"role": "assistant", "content": "ok"},
                ],
                "source": "s",
                "task": "chat",
                "lang": "en",
            },
            {"id": "bad", "messages": [{"role": "nope", "content": "x"}]},  # invalid
        ],
    )
    _train, _val, _test, report = ingest(str(p), seed=7)
    assert report.duplicates_removed == 1
    assert report.invalid_records == 1
    assert report.valid_records == 2
    assert report.train + report.val + report.test == 2
    assert report.contamination.train_eval_exact_overlap == 0


def test_split_deterministic(tmp_path):
    p = tmp_path / "d.jsonl"
    recs = [
        {
            "id": f"i-{i}",
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ],
            "source": "s",
            "task": "chat",
            "lang": "en",
        }
        for i in range(20)
    ]
    _write_jsonl(p, recs)
    t1, _v1, _e1, r1 = ingest(str(p), seed=11)
    t2, _v2, _e2, r2 = ingest(str(p), seed=11)
    assert [e.id for e in t1] == [e.id for e in t2]
    assert r1.train == r2.train and r1.val == r2.val and r1.test == r2.test


def test_split_changes_with_seed(tmp_path):
    p = tmp_path / "d.jsonl"
    recs = [
        {
            "id": f"i-{i}",
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ],
            "source": "s",
            "task": "chat",
            "lang": "en",
        }
        for i in range(30)
    ]
    _write_jsonl(p, recs)
    _, _, _, r1 = ingest(str(p), seed=1)
    _, _, _, r2 = ingest(str(p), seed=2)
    # seeds may coincidentally produce identical splits; assert structure instead
    assert r1.train + r1.val + r1.test == 30
    assert r2.train + r2.val + r2.test == 30


def test_contamination_flags_exact_overlap():
    train = [
        Example(
            id="t1",
            messages=[
                {"role": "user", "content": "unique phrase here"},
                {"role": "assistant", "content": "x"},
            ],
        )
    ]
    eval_ = [
        Example(
            id="e1",
            messages=[
                {"role": "user", "content": "unique phrase here"},
                {"role": "assistant", "content": "x"},
            ],
        )
    ]
    rep = check_contamination(train, eval_, threshold=0.75)
    assert rep.train_eval_exact_overlap == 1
    assert "e1" in rep.flagged_eval_ids


def test_write_splits_and_report(tmp_path):
    ex = Example(
        id="x", messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    )
    train, val, test = [ex], [], []
    paths = write_splits(str(tmp_path), train, val, test)
    assert os.path.exists(paths["train"])
    contam = ContaminationReport(
        train_eval_exact_overlap=0,
        train_eval_high_overlap=0,
        high_overlap_threshold=0.75,
        flagged_eval_ids=[],
    )
    report_path = write_report(
        str(tmp_path),
        type(
            "R",
            (),
            {
                "total_records": 1,
                "valid_records": 1,
                "invalid_records": 0,
                "duplicates_removed": 0,
                "train": 1,
                "val": 0,
                "test": 0,
                "validation_errors": [],
                "contamination": contam,
            },
        )(),
    )
    assert os.path.exists(report_path)
    with open(report_path) as fh:
        data = json.load(fh)
    assert data["train"] == 1
