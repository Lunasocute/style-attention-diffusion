import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
from transformers import CLIPImageProcessor
from config import Config

class StyleDiffusionDataset(Dataset):
    """
    Dataset for style-conditioned diffusion training.

    Each sample provides:
        - A target image  (used to compute VAE latents and noise prediction loss)
        - A style image   (fed into CLIP-Vision to produce style embeddings)
        - A blank text prompt (for unconditional training)

    The dataset supports optional "self-style" augmentation: with a given probability,
    the style image is replaced by the target image to encourage identity-style learning.
    """
    def __init__(self, csv_file, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        
        # Preprocessing for the target image (to be encoded by the VAE).
        # VAE expects images normalized to [-1, 1].
        self.img_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        
        # Preprocessing for the style image (to be encoded by CLIP-Vision).
        # Use the official CLIP image processor for normalization and resizing.
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        target_path = row['target_image']
        style_path = row['style_image']
        target_img = Image.open(target_path).convert("RGB")

        # With probability p, use the target image itself as the style image.
        # This helps stabilize training by anchoring style representations.
        if torch.rand(1).item() < Config.SELF_STYLE_PROB:
            style_img = target_img
        else:
            style_img = Image.open(style_path).convert("RGB")
        
        # 1. Target image for VAE (pixel values in [-1, 1])
        pixel_values = self.img_transform(target_img)
        
        # 2. Style image raw CLIP-Vision inputs.
        #    CLIPImageProcessor returns a batch dimension -> remove it.
        style_inputs = self.clip_processor(images=style_img, return_tensors="pt").pixel_values.squeeze(0)
        
        # 3. Blank text prompt for unconditional generation branch. Produces input_ids with shape [max_length].
        text_inputs = self.tokenizer(
            "", 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.squeeze(0)

        return {
            "pixel_values": pixel_values,                     # For VAE → latents
            "style_pixel_values": style_inputs,               # For CLIP-Vision → style tokens
            "input_ids": text_inputs                          # For unconditional text encoder
        }