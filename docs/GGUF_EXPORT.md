# GGUF export & quantization (llama.cpp workflow)

Kivra checkpoints trained here are **LoRA adapters only** by default. Two
distribution options exist:

1. **Adapter-only** — redistribute the small adapter directory; users must have
   access to the base Qwen checkpoint themselves. Smaller download, respects the
   base checkpoint's distribution terms.
2. **Merged weights** — merge adapters into the base and export a standalone
   GGUF that runs in llama.cpp without the base checkpoint. Larger download,
   but simplest for end users.

## Step 1 — merge adapters

```bash
python - <<'EOF'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen3-0.6B-Base"   # or your actual base
adapter = "runs/kivra-spark"

model = AutoModelForCausalLM.from_pretrained(base, device_map="cpu", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(base)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained("runs/kivra-spark/merged")
tokenizer.save_pretrained("runs/kivra-spark/merged")
EOF
```

## Step 2 — convert to GGUF (llama.cpp)

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp && cmake -B build -B build -DLLAMA_BUILD=ON && cmake --build build --target quantize -j

python3 ~/llama.cpp/convert-hf-to-gguf.py runs/kivra-spark/merged --outtype f16 \
  --outfile kivra-spark-merged.f16.gguf
```

## Step 3 — quantize

```bash
~/llama.cpp/build/bin/quantize kivra-spark-merged.f16.gguf \
  kivra-spark-0.6b-instruct-v1.Q4_K_M.gguf Q4_K_M
```

- `Q4_K_M` recommended for phones/PC (≈1 GB for 0.6B).
- `Q8_0` if quality matters more than size.

## Step 4 — run

```bash
~/llama.cpp/main -m kivra-spark-0.6b-instruct-v1.Q4_K_M.gguf -p "Hello, Kivra!" -n 128
```

Android: see `android/DEPLOYMENT.md` for the llama.cpp Android build path and
memory/context guidance.
