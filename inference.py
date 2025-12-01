import os
import torch
import sys
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from model import ImageProjModel, StyleAttnProcessor
from config import Config

USE_STYLE = True       # Toggle style guidance on/off (False = baseline SD, True = style-conditioned)

def load_style_attn_procs_to_unet(unet, device):
    """
    Inject StyleAttnProcessor into the UNet and load their trained weights.
    """
    # 1. Rebuild attn_procs
    attn_procs = {}
    for name, _ in unet.attn_processors.items():
        if name.endswith("attn2.processor"):
            layer_name = name.split(".processor")[0]
            curr_layer = unet.get_submodule(layer_name)
            hidden_size = curr_layer.to_k.out_features

            processor = StyleAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=unet.config.cross_attention_dim,
                num_style_tokens=Config.NUM_STYLE_TOKENS,
                scale=1.0,
            ).to(device)

            attn_procs[name] = processor
        else:
            # Keep attn1 as the default efficient processor
            attn_procs[name] = AttnProcessor2_0()

    # Replace all processors at once
    unet.set_attn_processor(attn_procs)

    # 2. Load saved StyleAttnProcessor weights
    ckpt_style = os.path.join(Config.OUTPUT_DIR, "style_attn_procs.pth")
    if not os.path.exists(ckpt_style):
        raise FileNotFoundError(f"Cannot find {ckpt_style}, have you trained & saved it?")

    style_state = torch.load(ckpt_style, map_location=device)

    for name, proc in unet.attn_processors.items():
        if isinstance(proc, StyleAttnProcessor) and name in style_state:
            proc.load_state_dict(style_state[name])

    print("Loaded StyleAttnProcessor weights from style_attn_procs.pth")


def inference(style_image_path, output_path="output.png"):
    device = Config.DEVICE

    # Set seeds before any pipeline/latent initialization for reproducibility
    torch.manual_seed(3211)
    torch.cuda.manual_seed_all(3211)

    # 1. Load base Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        Config.SD_MODEL_ID,
        safety_checker=None,       # disable safety checker
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.vae.eval()
    pipe.unet.eval()
    pipe.text_encoder.eval()

    # 2. Inject and load StyleAttnProcessor weights
    load_style_attn_procs_to_unet(pipe.unet, device)

    STYLE_SCALE = 0.5         # Try 0.0, 0.1, 0.3, 0.5, ...

    for proc in pipe.unet.attn_processors.values():
        if isinstance(proc, StyleAttnProcessor):
            proc.scale = STYLE_SCALE
            print("gate=", float(proc.gate.detach()), "scale=", proc.scale)

    # 3. Load trained ImageProjModel
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        Config.CLIP_IMAGE_MODEL
    ).to(device)
    image_encoder.eval()

    image_processor = CLIPImageProcessor.from_pretrained(Config.CLIP_IMAGE_MODEL)

    image_proj = ImageProjModel(
        input_dim=image_encoder.config.projection_dim,
        cross_attention_dim=pipe.unet.config.cross_attention_dim,
    ).to(device)
    ckpt_image_proj = os.path.join(Config.OUTPUT_DIR, "image_proj.pth")
    image_proj.load_state_dict(torch.load(ckpt_image_proj, map_location=device))
    image_proj.eval()

    # 4. Prepare style image → style_tokens
    style_img = Image.open(style_image_path).convert("RGB")
    inputs = image_processor(images=style_img, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        image_embeds = image_encoder(**inputs).image_embeds  # [1, 768]
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        style_tokens = image_proj(image_embeds)  # [1, 1, 768]

    # 5. Manual sampling loop (using combined text + style embeddings)
    # Latents for 512x512: [B, 4, 64, 64]
    latents = torch.randn((1, 4, 64, 64), device=device)
    pipe.scheduler.set_timesteps(50)

    # Empty text embedding (same as training using "" prompt)
    # encode_prompt returns (prompt_embeds, _); we only need the first
    null_prompt_embeds = pipe.encode_prompt("", device, 1, False)[0]  # [1, 77, 768]

    if USE_STYLE:
        # With style tokens
        combined_embeds = torch.cat([null_prompt_embeds, style_tokens], dim=1)
        null_style = torch.zeros_like(style_tokens)
        uncond_combined = torch.cat([null_prompt_embeds, null_style], dim=1)
    else:
        # Baseline: no style, only empty text
        combined_embeds = null_prompt_embeds
        uncond_combined = null_prompt_embeds
    batch_embeds = torch.cat([uncond_combined, combined_embeds], dim=0)

    guidance_scale = 8.0
    print("Generating with style guidance...")
    with torch.no_grad():
        for t in pipe.scheduler.timesteps:
            # Classifier-free guidance: [uncond, cond]
            latent_model_input = torch.cat([latents] * 2, dim=0)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = pipe.unet(
                latent_model_input, t, encoder_hidden_states=batch_embeds
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # 6. Decode latents → Image
    with torch.no_grad():
        image = pipe.decode_latents(latents)
        image = pipe.numpy_to_pil(image)[0]
        image.save(output_path)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    test_dir = "test"
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    args = sys.argv[1:]
    
    if len(args) == 0:
        if not os.path.isdir(test_dir):
            print(f"Folder '{test_dir}' not found, or pass image name in command.")
            sys.exit(1)
        for fname in os.listdir(test_dir):
            if not fname.lower().endswith(exts):
                continue

            input_path = os.path.join(test_dir, fname)
            base = fname.rsplit(".", 1)[0]
            output_path = os.path.join(test_dir, f"out_{base}.png")
            inference(input_path, output_path)

    input_image_name = args[0]
    base = input_image_name.rsplit(".", 1)[0]
    output_image_name = f"out_{base}.png"

    inference(input_image_name, output_image_name)