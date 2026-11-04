# Minimal local chat interface (REPL). No hosted API, no API keys.

from __future__ import annotations

import sys

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None


def _require_torch():
    if torch is None:
        raise SystemExit("torch is required: pip install -e '.[inference]'")


def chat_loop(
    base_model: str,
    *,
    adapter: str | None = None,
    system_prompt: str = "You are Kivra, a helpful assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    load_in_4bit: bool = False,
) -> None:
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

    device = next(model.parameters()).device
    history = [{"role": "system", "content": system_prompt}]
    print("Kivra local chat. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            print()
            break
        if user.lower() in ("exit", "quit"):
            break
        if not user:
            continue
        history.append({"role": "user", "content": user})

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{t['role']}: {t['content']}" for t in history) + "\nassistant:"

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
        reply = tokenizer.decode(
            out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        print(f"Kivra: {reply}\n")
        history.append({"role": "assistant", "content": reply})


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Kivra minimal local chat")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--system", default="You are Kivra, a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args(argv)
    _require_torch()
    chat_loop(
        args.base,
        adapter=args.adapter,
        system_prompt=args.system,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
