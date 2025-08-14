from pali_gemma.load_weight import load_hf_model


if __name__ == "__main__":
    model, tokenizer = load_hf_model("./models_weight")