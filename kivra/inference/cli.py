"""One-shot local inference: load base (+ optional LoRA adapter) and generate."""

from __future__ import annotations

import argparse
import sys

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None


def _require_torch():
    if torch is None:
        raise SystemExit("torch is required: pip install -e '.[inference]'")


def generate(
    base_model: str,
    *,
    adapter: str | None = None,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    load_in_4bit: bool = False,
    device_map: str = "auto",
) -> str:
    _require_torch()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=False,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Kivra local one-shot inference")
    parser.add_argument("--base", required=True, help="Base model repo id or path")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory (optional)")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args(argv)

    _require_torch()
    text = generate(
        args.base,
        adapter=args.adapter,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
    )
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
