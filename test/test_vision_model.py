from pali_gemma.config import ModelConfig
from pali_gemma.model.vision_model import SiglipVisionModel


import torch 

def main():
    config = ModelConfig()

    model = SiglipVisionModel(config)
    

    B = 8
    imgs = torch.randn(B, config.vision_num_channels, config.vision_image_size, config.vision_image_size)
    print(f"Input shape: {imgs.shape}"  )
    outputs = model(imgs)
    
    assert outputs.shape == (B, config.vision_num_image_tokens, config.vision_hidden_size), "Output shape mismatch"

    print("Pass Test ")
    

if __name__ == "__main__":
    main()
