#!/bin/bash

MODEL_PATH="./models_weight"
PROMPT="Question: What the image about?"
IMAGE_FILE_PATH="./cat.png"
ANSWER="It is a cute cat doing the programming"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LR=1e-4
EPOCHS=10
STEPS=1000
SAVE_ADAPTER_PATH="./lora_adapter"
ONLY_CPU="False"

python pali_gemma_ft.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --answer "$ANSWER" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lr $LR \
    --epochs $EPOCHS \
    --steps $STEPS \
    --save_adapter_path "$SAVE_ADAPTER_PATH" \
    --only_cpu $ONLY_CPU \