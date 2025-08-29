#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
MODEL_PATH="./models_weight"
PROMPT="Extract JSON"
IMAGE_FILE_PATH="./receipt.png"
MAX_TOKENS_TO_GENERATE=50
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"     # "True" or "False"
ONLY_CPU="False"     # "True" or "False"

# ---- LoRA (optional) ----
USE_LORA="True"                 # "True" to enable
LORA_NAME="cord_lora"   # name of the adapter saved with save_adapter(...)
LORA_MERGE="False"               # "True" to merge into base weights


# Run
python inference.py \
  --model_path "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --image_file_path "$IMAGE_FILE_PATH" \
  --max_tokens_to_generate "$MAX_TOKENS_TO_GENERATE" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --do_sample "$DO_SAMPLE" \
  --only_cpu "$ONLY_CPU" \
  --lora_adapter_name "$LORA_NAME" \
  --lora_merge "$LORA_MERGE"

