import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.autoencoding_utils.encoder_utils import get_2d_sincos_pos_embed

ACT2FN = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}

class DecoderConfig:
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=256,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_intermediate_size=2048,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size


class DecoderLayer(nn.Module):
    """Single transformer block: pre-norm self-attention + FFN with residuals."""

    def __init__(self, hidden_size, num_attention_heads, intermediate_size,
                 hidden_act="gelu", hidden_dropout=0.0, attn_dropout=0.0,
                 layer_norm_eps=1e-12, qkv_bias=True):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.attn_dropout_p = attn_dropout

        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(hidden_dropout)

        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_up = nn.Linear(hidden_size, intermediate_size)
        self.ffn_act = ACT2FN[hidden_act] if isinstance(hidden_act, str) else hidden_act
        self.ffn_down = nn.Linear(intermediate_size, hidden_size)
        self.ffn_drop = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states):
        # Self-attention
        x = self.layernorm_before(hidden_states)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout_p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.attn_proj(x))
        hidden_states = x + hidden_states

        # FFN
        x = self.layernorm_after(hidden_states)
        x = self.ffn_down(self.ffn_act(self.ffn_up(x)))
        x = self.ffn_drop(x)
        hidden_states = x + hidden_states

        return hidden_states


class Decoder(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.config = config
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )

        # Transformer layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=config.decoder_hidden_size,
                num_attention_heads=config.decoder_num_attention_heads,
                intermediate_size=config.decoder_intermediate_size,
                hidden_act=config.hidden_act,
                hidden_dropout=config.hidden_dropout_prob,
                attn_dropout=config.attention_probs_dropout_prob,
                layer_norm_eps=config.layer_norm_eps,
                qkv_bias=config.qkv_bias,
            )
            for _ in range(config.decoder_num_hidden_layers)
        ])

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size ** 2 * config.num_channels, bias=True
        )

        # Trainable CLS token for the decoder
        self.trainable_cls_token = nn.Parameter(torch.zeros(config.decoder_hidden_size))

        self.gradient_checkpointing = False
        self._initialize_weights()    

    def _initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True,
            extra_tokens=1,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def interpolate_latent(self, x):
        """Bilinear interpolation if latent length doesn't match num_patches."""
        b, l, c = x.shape
        if l == self.num_patches:
            return x
        h = w = int(l ** 0.5)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        target = int(self.num_patches ** 0.5)
        x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous().view(b, self.num_patches, c)

    def interpolate_pos_encoding(self, embeddings):
        """Bicubic interpolation of decoder position embeddings if lengths differ."""
        n_embed = embeddings.shape[1] - 1
        n_pos = self.decoder_pos_embed.shape[1] - 1
        if n_embed == n_pos:
            return self.decoder_pos_embed

        cls_pos = self.decoder_pos_embed[:, 0:1, :]
        patch_pos = self.decoder_pos_embed[:, 1:, :]
        dim = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, 1, -1, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            scale_factor=(1, n_embed / n_pos),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def unpatchify(self, patchified_pixel_values):
        """Reshape [B, N, patch_size^2 * C] → [B, C, H, W]."""
        ps = self.config.patch_size
        nc = self.config.num_channels
        h = w = int(patchified_pixel_values.shape[1] ** 0.5)
        x = patchified_pixel_values.reshape(-1, h, w, ps, ps, nc)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(-1, nc, h * ps, w * ps)    

    def forward(self, hidden_states, drop_cls_token=False, interpolate_pos_encoding=False):
        """
        Args:
            hidden_states: [B, L, encoder_hidden_size] from encoder.
            drop_cls_token: If True, remove first token before interpolation.
            interpolate_pos_encoding: If True, bicubic-interpolate position embeddings.

        Returns:
            logits: [B, num_patches, patch_size^2 * num_channels]
        """
        x = self.decoder_embed(hidden_states)

        # Optionally drop CLS token, then interpolate latent to match num_patches
        if drop_cls_token:
            x = self.interpolate_latent(x[:, 1:, :])
        else:
            x = self.interpolate_latent(x)

        # Prepend trainable CLS token
        cls_token = self.trainable_cls_token[None, None].expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Add position embeddings
        if interpolate_pos_encoding:
            assert drop_cls_token, "interpolate_pos_encoding only works with drop_cls_token=True"
            x = x + self.interpolate_pos_encoding(x)
        else:
            x = x + self.decoder_pos_embed

        # Transformer blocks
        for layer in self.decoder_layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.decoder_norm(x)
        logits = self.decoder_pred(x)

        # Remove CLS token
        return logits[:, 1:, :]