"""Schema for a single chat training example.

Fields
------
id : str
    Unique record id.
messages : list[dict]
    ChatML-style turns, each ``{"role": "system"|"user"|"assistant", "content": str}``.
source : str
    Origin of the record (dataset name, file, demo flag, etc.).
task : str | None
    Task family hint (e.g. "chat", "code", "summarize").
lang : str
    BCP-47 language hint, default "en".
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_ROLES = {"system", "user", "assistant"}


@dataclass
class Example:
    id: str
    messages: list[dict]
    source: str = "unknown"
    task: str | None = None
    lang: str = "en"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.id, str) or not self.id.strip():
            errors.append("id must be a non-empty string")
        if not isinstance(self.messages, list) or len(self.messages) == 0:
            errors.append("messages must be a non-empty list")
        else:
            for i, turn in enumerate(self.messages):
                if not isinstance(turn, dict):
                    errors.append(f"messages[{i}] must be an object")
                    continue
                role = turn.get("role")
                content = turn.get("content")
                if role not in VALID_ROLES:
                    errors.append(f"messages[{i}].role must be one of {sorted(VALID_ROLES)}")
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"messages[{i}].content must be a non-empty string")
        if self.task is not None and not isinstance(self.task, str):
            errors.append("task must be a string or None")
        if not isinstance(self.lang, str) or not self.lang:
            errors.append("lang must be a non-empty string")
        return errors

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "messages": self.messages,
            "source": self.source,
            "task": self.task,
            "lang": self.lang,
        }


@dataclass
class ValidationError:
    line: int
    record_id: str
    errors: list[str]


def validate_example(raw: dict, line: int) -> tuple[Example, list[ValidationError]]:
    record_id = raw.get("id", f"line-{line}") if isinstance(raw, dict) else f"line-{line}"
    errors: list[ValidationError] = []
    if not isinstance(raw, dict):
        errors.append(
            ValidationError(
                line=line, record_id=str(record_id), errors=["record must be a JSON object"]
            )
        )
        return Example(id=str(record_id), messages=[]), errors
    example = Example(
        id=str(raw.get("id", f"line-{line}")),
        messages=raw.get("messages", []),
        source=raw.get("source", "unknown"),
        task=raw.get("task"),
        lang=raw.get("lang", "en"),
    )
    field_errors = example.validate()
    if field_errors:
        errors.append(ValidationError(line=line, record_id=example.id, errors=field_errors))
    return example, errors
