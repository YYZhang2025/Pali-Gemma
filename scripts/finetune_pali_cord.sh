#!/bin/bash
# ---- Paths ----
MODEL_PATH="./models_weight"
ADAPTERS_DIR="./lora_adapters"
SAVE_ADAPTER_PATH="cord_lora"

# ---- Prompt (teacher forcing prefix) ----
PROMPT="extract JSON. "

# ---- Dataset (CORD, Donut format) ----
DATASET_NAME_OR_PATH="naver-clova-ix/cord-v2"
TRAIN_SPLIT="train"
VAL_SPLIT="validation"   # change to "test" if your build has no validation split
BATCH_SIZE=1
NUM_WORKERS=2

# ---- LoRA ----
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05

# ---- Training ----
LR=1e-4
EPOCHS=1
GRAD_ACCUM=8
ONLY_CPU="False"

python finetune_cord.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lr $LR \
    --epochs $EPOCHS \
    --grad_accum $GRAD_ACCUM \
    --dataset_name_or_path "$DATASET_NAME_OR_PATH" \
    --train_split "$TRAIN_SPLIT" \
    --val_split "$VAL_SPLIT" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --adapters_dir "$ADAPTERS_DIR" \
    --save_adapter_path "$SAVE_ADAPTER_PATH" \
    --only_cpu $ONLY_CPU
