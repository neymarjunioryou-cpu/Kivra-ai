#!/usr/bin/env bash
# Export a fine-tuned Kivra adapter to a quantized GGUF for llama.cpp.
#
# Workflow:
#   1. Merge LoRA adapters into the base model (merged weights, fp16).
#   2. Convert merged HF weights to llama.cpp GGUF format.
#   3. Quantize to Q4_K_M (recommended for phones/PC) or Q8_0.
#
# Nothing here downloads or trains anything by itself.

set -euo pipefail

usage() {
  echo "Usage: $0 <merged-model-dir> <output-gguf> [quant-type=Q4_K_M]"
  echo "Example: $0 runs/kivra-spark/merged kivra-spark-0.6b.Q4_K_M.gguf Q4_K_M"
  exit 1
}

MERGED_DIR="${1:-}"
OUTPUT_GGUF="${2:-}"
QUANT="${3:-Q4_K_M}"

if [[ -z "$MERGED_DIR" || -z "$OUTPUT_GGUF" ]]; then usage; fi

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"

echo "[1/3] Merge adapters into base model (Python script: kivra/export/merge_and_save.py)"
echo "    See kivra/export/merge_and_save.py --adapter <adapter> --base <base> --out $MERGED_DIR"

echo "[2/3] Convert to GGUF (unquantized)"
if [[ ! -x "$LLAMA_CPP_DIR/convert-hf-to-gguf.py" ]]; then
  echo "  Clone llama.cpp first: git clone https://github.com/ggerganov/llama.cpp $LLAMA_CPP_DIR"
fi
python3 "$LLAMA_CPP_DIR/convert-hf-to-gguf.py" "$MERGED_DIR" \
  --outtype f16 --outfile "${OUTPUT_GGUF%.gguf}.f16.gguf"

echo "[3/3] Quantize to $QUANT"
if [[ ! -x "$LLAMA_CPP_DIR/quantize" ]]; then
  echo "  Build quantize: cmake -B $LLAMA_CPP_DIR/build -S $LLAMA_CPP_DIR -DLLAMA_BUILD=ON && cmake --build $LLAMA_CPP_DIR/build --target quantize -j"
fi
"$LLAMA_CPP_DIR/build/bin/quantize" "${OUTPUT_GGUF%.gguf}.f16.gguf" "$OUTPUT_GGUF" "$QUANT"

echo "Done: $OUTPUT_GGUF"
echo "Adapter-only vs merged: adapters are small and redistributable without the base "
echo "weights; merged GGUF contains full weights and can run standalone in llama.cpp."
