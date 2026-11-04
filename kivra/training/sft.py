"""Supervised fine-tuning trainer for Kivra profiles.

Implements LoRA / QLoRA SFT over Hugging Face Transformers + PEFT. Torch and
friends are imported lazily so ``kivra.training`` can be imported in dry-run
mode without downloading anything.
"""

from __future__ import annotations

import json
import os

from ..config import Profile, load_profile
from ..data.ingest import ingest, write_report, write_splits


class MissingTrainingDeps(RuntimeError):
    """Raised when torch/transformers/peft are not installed."""


def _require_training_deps():
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingTrainingDeps(
            "Training dependencies are not installed. Run:\n"
            "  pip install -e '.[train]'\n"
            "Original error: " + str(exc)
        ) from exc


def _format_messages(messages: list[dict], tokenizer) -> str:
    """Apply the tokenizer chat template when available, else a simple fallback."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    parts = []
    for turn in messages:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts) + "\nassistant:"


def train(
    profile: Profile,
    *,
    data_path: str,
    output_dir: str | None = None,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
) -> dict:
    """Run SFT for the given profile.

    Returns a small info dict (never invents metrics). If ``dry_run`` is True,
    no weights are downloaded and no training happens — it validates config,
    data, and the pipeline shape instead.
    """
    if output_dir is None:
        output_dir = profile.train.output_dir

    # --- Data: validate + split (works without torch) ---------------------
    train_ex, val_ex, test_ex, report = ingest(data_path)
    split_dir = os.path.join(output_dir, "data_splits")
    write_splits(split_dir, train_ex, val_ex, test_ex)
    write_report(split_dir, report)

    info = {
        "profile": profile.name,
        "base_model": profile.model.base_model,
        "output_dir": output_dir,
        "train_examples": len(train_ex),
        "val_examples": len(val_ex),
        "test_examples": len(test_ex),
        "dry_run": dry_run,
        "ingest_report": split_dir,
    }

    if dry_run:
        info["status"] = "dry-run-ok"
        return info

    # --- Real training (torch required) -----------------------------------
    _require_training_deps()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    os.makedirs(output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        profile.model.base_model,
        revision=profile.model.revision,
        trust_remote_code=profile.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization (QLoRA) or plain load
    bnb_config = None
    if profile.quantization.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(
                torch, profile.quantization.bnb_4bit_compute_dtype, torch.float16
            ),
            bnb_4bit_quant_type=profile.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=profile.quantization.bnb_4bit_use_double_quant,
        )

    model = AutoModelForCausalLM.from_pretrained(
        profile.model.base_model,
        revision=profile.model.revision,
        trust_remote_code=profile.model.trust_remote_code,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if profile.quantization.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if profile.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=profile.lora.r,
        lora_alpha=profile.lora.lora_alpha,
        lora_dropout=profile.lora.lora_dropout,
        target_modules=profile.lora.target_modules,
        bias=profile.lora.bias,
        task_type=profile.lora.task_type,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset
    class ChatDataset(torch.utils.data.Dataset):
        def __init__(self, examples, tokenizer, max_length):
            self.items = []
            for ex in examples:
                text = _format_messages(ex.messages, tokenizer)
                enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
                self.items.append({k: v.squeeze(0) for k, v in enc.items()})

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    train_ds = ChatDataset(train_ex, tokenizer, profile.train.max_seq_length)
    val_ds = ChatDataset(val_ex, tokenizer, profile.train.max_seq_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=profile.train.per_device_train_batch_size,
        shuffle=True,
        num_workers=profile.train.dataloader_num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=profile.train.per_device_train_batch_size,
        shuffle=False,
        num_workers=profile.train.dataloader_num_workers,
    )

    # Optimizer / scheduler
    from transformers import get_scheduler

    optimizer = torch.optim.AdamW(model.parameters(), lr=profile.train.learning_rate)
    num_training_steps = (
        profile.train.max_steps
        if profile.train.max_steps > 0
        else len(train_loader) * profile.train.num_train_epochs
    )
    scheduler = get_scheduler(
        profile.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * profile.train.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    global_step = 0
    best_val_loss = float("inf")
    for epoch in range(profile.train.num_train_epochs):
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],
            )
            loss = outputs.loss / profile.train.gradient_accumulation_steps
            loss.backward()
            running += loss.item() * profile.train.gradient_accumulation_steps

            if (global_step + 1) % profile.train.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            steps_seen = global_step + 1
            if global_step % profile.train.logging_steps == 0:
                print(f"epoch {epoch} step {global_step} train_loss {running / steps_seen:.4f}")

            if (global_step + 1) % profile.train.save_steps == 0:
                ckpt = os.path.join(output_dir, f"checkpoint-{global_step + 1}")
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"checkpoint saved: {ckpt}")

            if profile.train.max_steps > 0 and global_step + 1 >= profile.train.max_steps:
                break
            global_step += 1

        # Validation loss (measured, not reported as a benchmark score)
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"],
                )
                val_loss += outputs.loss.item()
                val_steps += 1
        model.train()
        avg_val = val_loss / max(val_steps, 1)
        print(f"epoch {epoch} val_loss {avg_val:.4f}")
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(os.path.join(output_dir, "best"))

    # Final adapter save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "training_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    info["status"] = "trained"
    info["best_val_loss"] = best_val_loss if best_val_loss != float("inf") else None
    info["adapters_only"] = True
    return info


def load_profile_entry(profile_name_or_path: str) -> Profile:
    """Accept a preset name ('spark'|'voltage') or a path to a YAML config."""
    if profile_name_or_path in ("spark", "voltage"):
        return load_profile(name=profile_name_or_path)
    return load_profile(path=profile_name_or_path)
