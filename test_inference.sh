#!/bin/bash

MODEL_PATH="./models_weight_mixed"
PROMPT="Question: What the image about? Answer: "
IMAGE_FILE_PATH="./cat.png"
MAX_TOKENS_TO_GENERATE=50
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="True"
ONLY_CPU="False"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \