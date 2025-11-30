import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.optimization import get_scheduler
import os
from config import Config
from dataset_loader import StyleDiffusionDataset
from model import ImageProjModel, StyleAttnProcessor

import torchvision.models as models
import matplotlib.pyplot as plt


# VGG Gram Matrix Loss Helper for Evaluation
def calc_gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(h * w)

class VGGStyleFeatures(torch.nn.Module):
    """
    Use VGG19 to extract multi-layer features for Gram Matrix style loss.

    We keep layers up to roughly relu4_1 (features[:21]) and select several
    intermediate layers as style layers.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        # Only keep the first few layers
        self.slice = torch.nn.Sequential(*list(vgg.children())[:21])
        for p in self.parameters():
            p.requires_grad = False

        # Indices of layers from which we take style features (empirical choice)
        self.style_layers = {1, 6, 11, 20}

    def forward(self, x):
        """
        x: [B, 3, H, W], values in [0, 1]
        Returns: list of feature maps at selected style layers.
        """
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        feats = []
        h = x
        for i, layer in enumerate(self.slice):
            h = layer(h)
            if i in self.style_layers:
                feats.append(h)
        return feats


def compute_style_loss(vgg, gen_imgs, ref_imgs):
    """
    Compute Gram Matrix style loss using VGG features.

    - vgg: VGGStyleFeatures module
    - gen_imgs: [B, 3, H, W] generated images, in [0, 1]
    - ref_imgs: [B, 3, H, W] reference style images, in [0, 1]
    """
    device = next(vgg.parameters()).device
    gen_imgs = gen_imgs.to(device)
    ref_imgs = ref_imgs.to(device)

    gen_feats = vgg(gen_imgs)
    ref_feats = vgg(ref_imgs)

    style_loss = 0.0
    for gf, rf in zip(gen_feats, ref_feats):
        gram_g = calc_gram_matrix(gf)
        gram_r = calc_gram_matrix(rf)
        style_loss = style_loss + F.mse_loss(gram_g, gram_r)

    return style_loss


def evaluate(unet, vae, image_proj, image_encoder, text_encoder,
             noise_scheduler, vgg, eval_batch, global_step, device):
    """
    Evaluation step (style consistency only).

    1. Use a style reference to generate an image (short DDPM sampling).
    2. Compute VGG Gram style loss between generated image and style reference.
    """
    unet.eval()
    vae.eval()
    image_encoder.eval()
    text_encoder.eval()

    with torch.no_grad():
        # --- 1. Take one eval sample --- 
        style_pixels = eval_batch["style_pixel_values"].to(device)
        input_ids = eval_batch["input_ids"].to(device)

        # Only use a single sample to save memory
        style_pixels = style_pixels[:1]
        input_ids = input_ids[:1]
        bsz = style_pixels.shape[0]

        gen = torch.Generator(device=device).manual_seed(1234)

        # --- 2. Simple DDPM sampling loop ---
        latents = torch.randn((bsz, 4, 64, 64), device=device, generator=gen)
        eval_scheduler = DDPMScheduler.from_config(noise_scheduler.config)
        eval_scheduler.set_timesteps(30)

        # Text + style tokens
        enc_text = text_encoder(input_ids)[0]                    # [B,77,768]
        image_embeds = image_encoder(style_pixels).image_embeds  # [B,768]
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)    

        style_tokens = image_proj(image_embeds)                  # [B, num_tokens, 768]
        cond = torch.cat([enc_text, style_tokens], dim=1)        # [B, 77 + num_tokens, 768]

        for t in eval_scheduler.timesteps:
            lat_in = eval_scheduler.scale_model_input(latents, t)
            noise_pred = unet(lat_in, t, encoder_hidden_states=cond).sample
            latents = eval_scheduler.step(noise_pred, t, latents).prev_sample

        # --- 3. Decode latents to images ---
        decoded = vae.decode(latents / 0.18215).sample
        gen_imgs = (decoded.clamp(-1, 1) + 1) / 2.0

        ref_imgs = (style_pixels.clamp(-1, 1) + 1) / 2.0

        # Downsample to 256×256 for style loss to save memory
        gen_imgs = F.interpolate(gen_imgs, size=(256, 256), mode="bilinear", align_corners=False)
        ref_imgs = F.interpolate(ref_imgs, size=(256, 256), mode="bilinear", align_corners=False)

        # --- 4. Compute style loss ---
        style_loss = compute_style_loss(vgg, gen_imgs, ref_imgs).item()

    unet.train()
    return style_loss


def save_checkpoint(image_proj, unet, output_dir, step=None):
    """
    Save the current ImageProjModel weights and all StyleAttnProcessor weights.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_proj_path = os.path.join(output_dir, "image_proj.pth")
    style_path = os.path.join(output_dir, "style_attn_procs.pth")

    torch.save(image_proj.state_dict(), image_proj_path)

    style_attn_state = {}
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, StyleAttnProcessor):
            style_attn_state[name] = proc.state_dict()

    torch.save(style_attn_state, style_path)

    print(f"✔ Saved checkpoint to: {output_dir}  (step={step})")


def main():
    # 1. Initialize core SD components
    noise_scheduler = DDPMScheduler.from_pretrained(Config.SD_MODEL_ID, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(Config.SD_MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(Config.SD_MODEL_ID, subfolder="text_encoder").to(Config.DEVICE)
    vae = AutoencoderKL.from_pretrained(Config.SD_MODEL_ID, subfolder="vae").to(Config.DEVICE)
    unet = UNet2DConditionModel.from_pretrained(Config.SD_MODEL_ID, subfolder="unet").to(Config.DEVICE)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(Config.CLIP_IMAGE_MODEL).to(Config.DEVICE)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # VGG for style loss (evaluation and optionally training)
    vgg = VGGStyleFeatures().to(Config.DEVICE)
    vgg.eval()

    # 2. Freeze base models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # 3. Initialize custom modules
    image_proj = ImageProjModel(
        input_dim=image_encoder.config.projection_dim,       
        cross_attention_dim=unet.config.cross_attention_dim, 
        hidden_dim=1024,
        num_tokens=Config.NUM_STYLE_TOKENS,
    ).to(Config.DEVICE)

    # Inject custom attention processors
    attn_procs = {}
    trainable_params = list(image_proj.parameters())

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
            )
            processor.to(Config.DEVICE)

            # Train the whole processor
            trainable_params += list(processor.parameters())
            attn_procs[name] = processor

        else:
            attn_procs[name] = AttnProcessor2_0()

    unet.set_attn_processor(attn_procs)

    # Try to resume from checkpoint if available
    ckpt_image_proj = os.path.join(Config.OUTPUT_DIR, "image_proj.pth")
    ckpt_style = os.path.join(Config.OUTPUT_DIR, "style_attn_procs.pth")

    if os.path.exists(ckpt_image_proj) and os.path.exists(ckpt_style):
        print(">>> Found checkpoints, loading weights...")

        image_proj_state = torch.load(ckpt_image_proj, map_location=Config.DEVICE)
        image_proj.load_state_dict(image_proj_state)
        print("Loaded image_proj.pth")

        style_attn_state = torch.load(ckpt_style, map_location=Config.DEVICE)

        for name, proc in unet.attn_processors.items():
            if isinstance(proc, StyleAttnProcessor) and name in style_attn_state:
                proc.load_state_dict(style_attn_state[name])

        print("Loaded style_attn_procs.pth")
        print(">>> Resume training from previous checkpoint.\n")
    else:
        print(">>> No checkpoint found, start training from scratch.\n")

    # 4. Dataloader
    dataset = StyleDiffusionDataset(Config.PAIRS_FILE, tokenizer)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Fixed eval batch
    eval_batch_size = min(4, Config.BATCH_SIZE)
    eval_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
    eval_batch = next(iter(eval_loader))

    # 5. Optimizer + Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=Config.LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-2)

    num_training_steps = Config.NUM_EPOCHS * len(dataloader)
    # num_training_steps = 200                           #debug
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * num_training_steps),
        num_training_steps=num_training_steps,
    )
    print(f"Using cosine LR scheduler: total_steps={num_training_steps}, warmup={int(0.05 * num_training_steps)}")

    style_loss_log = []
    eval_every = Config.EVAL_INTERVAL

    # 6. Training loop
    global_step = 0
    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch {epoch} started.")
        unet.train()
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(Config.DEVICE)         # Target Image
            style_pixels = batch["style_pixel_values"].to(Config.DEVICE)   # Style Ref
            input_ids = batch["input_ids"].to(Config.DEVICE)               # Empty Text
            
            # Encode target image to latents
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=Config.DEVICE
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Text embeddings (empty prompt)
            encoder_hidden_states_text = text_encoder(input_ids)[0]  # [B, 77, 768]

            # Style embeddings (CLIP-Vision)
            with torch.no_grad():
                image_embeds = image_encoder(style_pixels).image_embeds  # [B, 768]
                image_embeds = F.normalize(image_embeds, p=2, dim=-1)

            # Condition dropout (style dropout)
            if torch.rand(1) < Config.STYLE_DROPOUT_PROB:
                image_embeds = torch.zeros_like(image_embeds)

            # Project style embeddings into style tokens
            style_tokens = image_proj(image_embeds)  # [B, num_tokens, 768]

            # Concatenate text and style tokens
            combined_embeddings = torch.cat([encoder_hidden_states_text, style_tokens], dim=1)

            # --- Forward UNet ---
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=combined_embeddings
            ).sample

            # --- Noise prediction loss ---
            loss_mse = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = loss_mse
            print(
                f"Step {global_step}, loss = {loss.item():.4f}, "
            )
            global_step += 1

            # --- Backward ---
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # --- Periodic evaluation & checkpoint ---
            if global_step % eval_every == 0:
                save_checkpoint(image_proj, unet, Config.OUTPUT_DIR, step=global_step)
                eval_style_loss = evaluate(
                    unet=unet,
                    vae=vae,
                    image_proj=image_proj,
                    image_encoder=image_encoder,
                    text_encoder=text_encoder,
                    noise_scheduler=noise_scheduler,
                    vgg=vgg,
                    eval_batch=eval_batch,
                    global_step=global_step,
                    device=Config.DEVICE,
                )
                style_loss_log.append((global_step, eval_style_loss))
                print(f"[Eval] Step {global_step}, style_loss = {eval_style_loss:.4f}")

            if global_step >= num_training_steps:
                break    
            
    # Final checkpoint
    save_checkpoint(image_proj, unet, Config.OUTPUT_DIR)

    # Plot style loss curve if we collected any evals
    if len(style_loss_log) > 0:
        steps_style, losses_style = zip(*style_loss_log)
        plt.figure()
        plt.plot(steps_style, losses_style, label="Style Gram Loss", color='blue')
        plt.xlabel("Step")
        plt.ylabel("Style Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, "style_loss_curve.png"))
        print(f"Saved style loss plot to {os.path.join(Config.OUTPUT_DIR, 'style_loss_curve.png')}")

if __name__ == "__main__":
    main()