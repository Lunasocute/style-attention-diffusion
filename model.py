import torch
import torch.nn as nn
import torch.nn.functional as F

# Image Projector
# project CLIP Image Embedding into [B, num_tokens, D] that attention processor could use
class ImageProjModel(nn.Module):
    def __init__(self, input_dim=768, cross_attention_dim=768, hidden_dim=1024, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim

        # normalize CLIP embedding, mean->0, variance->1
        self.ln_pre = nn.LayerNorm(input_dim)    

        # Transformer MLP: Linear → GELU → Linear
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * cross_attention_dim),
        )

        
    def forward(self, image_embeds):
        x = self.ln_pre(image_embeds)                 # [B, input_dim]
        x = self.proj(x)                              # [B, cross_attention_dim]
        B = x.size(0)
        D = x.size(-1) // self.num_tokens
        x = x.view(B, self.num_tokens, D)             # [B, num_tokens, D]
        return x                                      # [B, num_tokens, D]


class StyleAttnProcessor(nn.Module):
    """
    Custom attention processor that injects *style tokens* into UNet cross-attention.

    Core idea:
    - Queries (Q) always come from UNet hidden_states (image latent features).
    - Keys/Values (K/V) come from two sources:
        (1) Text tokens   → processed by the original UNet attention weights.
        (2) Style tokens  → processed by custom projection layers.
    - Final output = text_attn_output + gate * style_attn_output.

    This module replaces the internal logic of CrossAttention in Diffusers UNet.
    """
    def __init__(self, hidden_size: int, cross_attention_dim: int, num_style_tokens: int = 4, scale: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size                      # UNet hidden dimension (C)
        self.cross_attention_dim = cross_attention_dim      # Dim of text/style tokens (D)
        self.num_style_tokens = num_style_tokens
        self.scale = scale                                  # global multiplier for style effect

        # Style-specific K/V projections: (D -> C)
        self.to_k_style = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_style = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        # Projection applied to style attention output
        self.style_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.style_out.weight.data.zero_()

        self.gate = nn.Parameter(torch.tensor(0.1))         # Trainable gate controlling style strength

    def _reshape_heads(self, x, batch_size, heads, head_dim):
        """
        Convert [B, L, C] → [B, heads, L, head_dim].
        Required for multi-head attention.
        """
        return x.view(batch_size, -1, heads, head_dim).transpose(1, 2)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        """
        Main attention hook that overrides CrossAttention logic.

        Args:
            attn:
                The original CrossAttention module from UNet.
                Provides:
                    - attn.to_q, attn.to_k, attn.to_v
                    - attn.to_out (linear + dropout)
                    - attn.heads

            hidden_states: [B, L, C]
                Image latent tokens from the current UNet block. These produce the Queries (Q).

            encoder_hidden_states: [B, L_text + L_style, D]
                Concatenated text tokens (+ optional style tokens).
                Only the last `num_style_tokens` tokens are treated as style.

        Returns:
            hidden: [B, L, C]
                Updated image features after injecting text + style attention.
                Returned back into the UNet forward pass.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Split text vs style tokens
        if encoder_hidden_states is None:
            text_context = hidden_states               # fallback (rare)
            style_context = None
        else:
            if encoder_hidden_states.shape[1] > self.num_style_tokens:
                text_context = encoder_hidden_states[:, :-self.num_style_tokens, :]
                style_context = encoder_hidden_states[:, -self.num_style_tokens :, :]
            else:
                text_context = encoder_hidden_states
                style_context = None

        # Handle uncond branch where style token vectors are zeroed out
        if style_context is not None and torch.all(style_context == 0):
            style_context = None

        # 2. Queries from UNet image features
        query = attn.to_q(hidden_states)  # [B, L, C]

        # 3. Standard text attention (K/V from UNet weights)
        key_text = attn.to_k(text_context)    # [B, L_txt, C]
        value_text = attn.to_v(text_context)  # [B, L_txt, C]

        inner_dim = key_text.shape[-1]
        head_dim = inner_dim // attn.heads

        # Split into multi-heads
        query = self._reshape_heads(query, batch_size, attn.heads, head_dim)
        key_text = self._reshape_heads(key_text, batch_size, attn.heads, head_dim)
        value_text = self._reshape_heads(value_text, batch_size, attn.heads, head_dim)

       # Text attention
        hidden_text = F.scaled_dot_product_attention(query, key_text, value_text, dropout_p=0.0, is_causal=False)
        hidden_text = (hidden_text.transpose(1, 2).reshape(batch_size, seq_len, inner_dim))

        out_text = attn.to_out[0](hidden_text)   # original output projection

        # 4. Style attention
        if style_context is not None:
            # Custom K/V projections for style tokens
            key_style = self.to_k_style(style_context)     # [B, L_style, C]
            value_style = self.to_v_style(style_context)   # [B, L_style, C]

            key_style = self._reshape_heads(key_style, batch_size, attn.heads, head_dim)
            value_style = self._reshape_heads(value_style, batch_size, attn.heads, head_dim)

            hidden_style = F.scaled_dot_product_attention(query, key_style, value_style, dropout_p=0.0, is_causal=False)
            hidden_style = (hidden_style.transpose(1, 2).reshape(batch_size, seq_len, inner_dim))

            # Style-specific projection
            out_style = self.style_out(hidden_style)  # [B, L, C]

            hidden = out_text + self.scale * self.gate  * out_style
        else:
            hidden = out_text

        # 5. Final dropout (same as original CrossAttention)
        hidden = attn.to_out[1](hidden)
        return hidden
