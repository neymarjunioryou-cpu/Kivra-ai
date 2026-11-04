"""Configuration dataclasses for Kivra Spark and Voltage training profiles.

All settings are optional overrides of the sample YAML files in ``configs/``.
Released model names must use the *actual* base parameter count — fine-tuning
does not change the size of a checkpoint (see README).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen3-0.6B-Base"
    revision: str | None = None
    trust_remote_code: bool = False


@dataclass
class QuantConfig:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainConfig:
    seed: int = 42
    output_dir: str = "runs/kivra-spark"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    max_steps: int = -1
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    max_seq_length: int = 1024
    packing: bool = False
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    report_to: list[str] = field(default_factory=list)


@dataclass
class Profile:
    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantConfig = field(default_factory=QuantConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ---- Preset profiles -----------------------------------------------------

SPARK_PROFILE = Profile(
    name="spark",
    model=ModelConfig(base_model="Qwen/Qwen3-0.6B-Base"),
    quantization=QuantConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
    lora=LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    ),
    train=TrainConfig(
        seed=42,
        output_dir="runs/kivra-spark",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2.0e-4,
        max_seq_length=1024,
        gradient_checkpointing=True,
    ),
)

VOLTAGE_PROFILE = Profile(
    name="voltage",
    model=ModelConfig(base_model="Qwen/Qwen2.5-3B-Base"),
    quantization=QuantConfig(
        load_in_4bit=False,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
    lora=LoRAConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    ),
    train=TrainConfig(
        seed=42,
        output_dir="runs/kivra-voltage",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=1.0e-4,
        max_seq_length=2048,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
    ),
)

PROFILES = {"spark": SPARK_PROFILE, "voltage": VOLTAGE_PROFILE}


def load_profile(path: str | None = None, name: str | None = None) -> Profile:
    """Load a Profile from a YAML file or a built-in preset name."""
    import yaml

    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return _dict_to_profile(raw)
    if name is not None:
        if name not in PROFILES:
            raise ValueError(f"Unknown profile '{name}'. Available: {sorted(PROFILES)}")
        return PROFILES[name]
    raise ValueError("Provide either --config PATH or --profile NAME.")


def _dict_to_profile(raw: dict) -> Profile:
    model = ModelConfig(**raw.get("model", {}))
    quantization = QuantConfig(**raw.get("quantization", {}))
    lora = LoRAConfig(**raw.get("lora", {}))
    train = TrainConfig(**raw.get("train", {}))
    return Profile(
        name=raw.get("profile", "custom"),
        model=model,
        quantization=quantization,
        lora=lora,
        train=train,
    )
