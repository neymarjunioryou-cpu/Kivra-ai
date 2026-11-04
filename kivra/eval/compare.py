"""Compare a base model against an adapted (LoRA/merged) model on held-out data.

Reports measured perplexity, generation latency (tokens/sec), and memory usage.
Nothing here is a benchmark score or an invented number.
"""

from __future__ import annotations

import json
import os
import time

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - torch optional for this module
    torch = None


class MissingInferenceDeps(RuntimeError):
    pass


def _require_torch():
    if torch is None:
        raise MissingInferenceDeps(
            "torch is required for evaluation. Run: pip install -e '.[inference]'"
        )


def _load_model_and_tokenizer(base_model: str, adapter: str | None, load_in_4bit: bool = False):
    _require_torch()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=False,
        device_map="auto" if torch.cuda.is_available() else None,
        load_in_4bit=load_in_4bit,
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def _perplexity(model, tokenizer, texts, max_length: int = 512, batch_size: int = 2) -> float:
    _require_torch()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=enc["input_ids"],
            )
        # mask out padding tokens
        labels = enc["input_ids"]
        pad = tokenizer.pad_token_id
        mask = (labels != pad).float()
        loss = out.loss.item()  # mean over non-pad tokens when labels given
        valid = mask.sum().item()
        total_nll += loss * valid
        total_tokens += valid
    if total_tokens == 0:
        return float("nan")
    return __import__("math").exp(total_nll / total_tokens)


def _latency(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> dict:
    _require_torch()
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = enc["input_ids"].shape[1]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start
    generated = out.shape[1] - input_len
    return {
        "input_tokens": input_len,
        "generated_tokens": generated,
        "elapsed_sec": round(elapsed, 4),
        "tokens_per_sec": round(generated / elapsed, 4) if elapsed > 0 else None,
    }


def compare(
    base_model: str,
    *,
    adapter: str | None = None,
    data_path: str | None = None,
    load_in_4bit: bool = False,
    out_path: str | None = None,
) -> dict:
    """Run base vs adapted comparison; returns measured metrics dict."""

    model, tokenizer = _load_model_and_tokenizer(base_model, adapter, load_in_4bit=load_in_4bit)

    metrics = {
        "base_model": base_model,
        "adapter": adapter,
        "load_in_4bit": load_in_4bit,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if data_path and os.path.exists(data_path):
        texts = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    texts.append(line)
        if texts:
            metrics["perplexity"] = round(_perplexity(model, tokenizer, texts), 4)
            metrics["eval_examples"] = len(texts)

    prompt = "The capital of Kenya is"
    metrics["latency"] = _latency(model, tokenizer, prompt)

    # Memory
    if torch.cuda.is_available():
        metrics["memory_gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 4)
    else:
        import resource

        metrics["memory_cpu_rss_gb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2), 4
        )

    print(json.dumps(metrics, indent=2))

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare base vs adapted Kivra model")
    parser.add_argument("--base", required=True, help="Base model repo id or path")
    parser.add_argument("--adapter", default=None, help="Adapter directory (optional)")
    parser.add_argument("--data", default=None, help="Held-out JSONL or text file")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--out", default=None, help="Write metrics JSON here")
    args = parser.parse_args()
    compare(
        args.base,
        adapter=args.adapter,
        data_path=args.data,
        load_in_4bit=args.load_in_4bit,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
