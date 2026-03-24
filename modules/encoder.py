import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Mlp

from modules.autoencoding_utils.encoder_utils import GaussianFourierEmbedding, LabelEmbedder, VisionRotaryEmbeddingFast, SwiGLUFFN, RMSNorm, NormAttention, get_2d_sincos_pos_embed, modulate


class Block(nn.Module):
    
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        block_norm=True,
        **block_kwargs,
    ):
        super().__init__()
        self.block_norm = block_norm
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            if self.block_norm: self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            if self.block_norm: self.norm3 = RMSNorm(hidden_size)
    
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )

        n_modulation = 4 if wo_shift else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, n_modulation * hidden_size, bias=True)
        )
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        if self.block_norm: x= self.norm3(x)
        return x


class FinalLayer(nn.Module):

    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class Encoder(nn.Module):
    
    def __init__(
        self,
        input_size=16,
        patch_size=1,
        in_channels=768,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_gembed: bool = True, 
        in_context_start=None,
        in_context_len=32,
        max_tokens=512,
        block_norm=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_gembed = use_gembed
        self.in_context_start = in_context_start
        self.in_context_len = in_context_len
        self.max_tokens = max_tokens
        
        self.up_sample = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.t_embedder = GaussianFourierEmbedding(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size), requires_grad=False)

        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim*2,
                pt_seq_len=self.max_tokens,
            )
                
            if self.in_context_start is not None:
                self.feat_rope_incontext = VisionRotaryEmbeddingFast(
                    dim=half_head_dim*2,
                    pt_seq_len=self.max_tokens,
                    num_cls_token=self.in_context_len
                )
                self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
                torch.nn.init.normal_(self.in_context_posemb, std=.02)
            else:
                self.in_context_start = torch.inf

        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            Block(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                     block_norm=block_norm,
                     ) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

        
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(256 ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    
    def forward(
        self, 
        x, 
        t=None, 
        y=None, 
        pos_embed=None, 
        checkpoint_blocks=False,
        img_patch_embed=None, 
        force_drop_ids_y_embedder=None
    ):

        x = self.up_sample(x)
        x = x + pos_embed

        if img_patch_embed is not None: x = torch.concat([x, img_patch_embed], dim=1)
        t = self.t_embedder(t)
        c = t
        y_emb = self.y_embedder(y, self.training, force_drop_ids=force_drop_ids_y_embedder)
        c = c + y_emb

        for block_idx, block in enumerate(self.blocks):
            if self.in_context_len > 0 and block_idx == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)

            rope = self.feat_rope if block_idx < self.in_context_start else self.feat_rope_incontext
            if checkpoint_blocks and self.training:
                # IMPORTANT: bind block/rope per-iteration for backward recomputation.
                # A plain lambda captures loop vars by reference and can replay the wrong block.
                def _run_block(x_in, c_in, _block=block, _rope=rope):
                    return _block(x_in, c_in, _rope)

                x = checkpoint(
                    _run_block,
                    x,
                    c,
                    use_reentrant=False,
                )
            else:
                x = block(x, c, rope)
        if self.in_context_start != torch.inf:
            x = x[:,self.in_context_len:]

        x = x[:,:256]
        x = self.final_layer(x, c)
        return x

    
    


    def forward_with_cfg(self, x, t, y, pos_embed, cfg_scale, bn_func=None, cfg_interval = (0.0, 1.0), cfg_norm_order="cfg_first"):
        """
        Forward pass, but also batches the unconditional forward pass for classifier-free guidance.

        Args:
            cfg_norm_order: Order of normalization and CFG. Options:
                - "cfg_first" (default): Apply CFG first, caller normalizes after
                - "norm_first": Normalize cond/uncond separately, then apply CFG
                - "both": Normalize before CFG (done here), caller also normalizes after
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, pos_embed=pos_embed)


        eps, rest = model_out[:, :, :self.in_channels], model_out[:, :, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        # Apply normalization before CFG if cfg_norm_order is "norm_first" or "both"
        if cfg_norm_order in ["norm_first", "both"] and bn_func is not None:
            cond_eps = bn_func(cond_eps)
            uncond_eps = bn_func(uncond_eps)

        low, high = cfg_interval
        interval_mask = (t < high) & ((low == 0.0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, cfg_scale, 1.0)
        # cfg_scale_interval has shape [2*N] (from t), but cond_eps/uncond_eps have shape [N]
        # Use only the first half corresponding to the conditional pass
        cfg_scale_interval_half = cfg_scale_interval[:len(cond_eps)]
        half_eps = uncond_eps + cfg_scale_interval_half.unsqueeze(-1).unsqueeze(-1) * (cond_eps - uncond_eps)
        
        eps = torch.cat([half_eps, half_eps], dim=0)
        # return torch.cat([eps, rest], dim=1)
        
        return_1 = torch.cat([eps, rest], dim=2)
        return return_1
