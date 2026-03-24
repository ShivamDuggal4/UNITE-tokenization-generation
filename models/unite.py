import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder import Encoder
from modules.decoder import Decoder, DecoderConfig
from modules.denoiser import Transport, Sampler
from modules.autoencoding_utils.lpips import LPIPS
from modules.autoencoding_utils.patch_embedding import ViTMAEEmbeddings
from modules.autoencoding_utils.encoder_utils import get_1d_sincos_pos_embed_from_grid


# Encoder model size: {name: (depth, hidden_size, num_heads)}
ENCODER_CONFIGS = {
    "Base": (12, 768, 12),
    "Medium": (12, 1024, 16),
    "Large": (24, 1024, 16),
    "XL": (28, 1152, 16),
}

# Decoder model size: {name: (num_layers, hidden_size, num_heads, mlp_size)}
DECODER_CONFIGS = {
    "Small": (12, 384, 6, 1536),
    "Base": (12, 768, 12, 3072),
    "Large": (24, 1024, 16, 4096),
}


class UNITE(nn.Module):

    def __init__(self, cfg,  num_latent_tokens=256, num_classes=1000):
        super().__init__()
        self.cfg = cfg    
        self.num_classes = num_classes
        self.num_latent_tokens = num_latent_tokens
        self.cfg_norm_order = str(cfg.get("cfg_norm_order", "norm_first"))
        if self.cfg_norm_order not in {"cfg_first", "norm_first", "both"}:
            raise ValueError(f"Invalid cfg_norm_order: {self.cfg_norm_order}")
        
        # --- Timestep shift ---
        self.timestep_shift_mode = cfg.get("timestep_shift_mode", "disabled")
        self._setup_timestep_shift(cfg)
        
        loss_cfg = cfg.get("loss", {})
        self.perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
        self.gen_loss_weight = float(loss_cfg.get("gen_loss_weight", 1.0))
        
        self.lpips_loss = LPIPS()
        self.lpips_loss.eval()


        self._build_encoder()
        self._build_decoder()
        self._build_denoiser(cfg)
        

    def _build_encoder(self):
        
        encoder_model_size = self.cfg.get("encoder_model_size", "Base")
        assert encoder_model_size in ENCODER_CONFIGS, f"Unknown encoder: {encoder_model_size}"
        encoder_depth, encoder_hidden_size, encoder_num_heads = ENCODER_CONFIGS[encoder_model_size]

        self.patch_size = int(self.cfg.get("patch_size", 16))
        self.diffusion_input_dim = int(self.cfg.get("diffusion_input_dim", 32))
        self.in_context_start = int(self.cfg.get("in_content_start", 999))
        self.in_context_start = None if self.in_context_start > 100 else self.in_context_start
        self.in_context_len = int(self.cfg.get("in_content_len", 32))

        self.noising_t_start = float(self.cfg.get("noising_t_start", 0.7))
        use_qknorm_encoder = bool(self.cfg.get("use_qknorm_encoder", False))
        self.modulation_recon_timestep_max = float(self.cfg.get("modulation_recon_timestep_max", 0.3))
        
        self.patch_embed = ViTMAEEmbeddings(
            image_size=256, patch_size=self.patch_size, num_channels=3,
            hidden_size=encoder_hidden_size,
        )
        self.num_image_patches = self.patch_embed.num_patches
        
        max_tokens = self.num_latent_tokens + self.num_image_patches
        self.encoder = Encoder(
            input_size=16, patch_size=1,
            in_channels=self.diffusion_input_dim,
            hidden_size=encoder_hidden_size,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=self.num_classes,
            learn_sigma=False,
            use_qknorm=use_qknorm_encoder, use_swiglu=True, use_rope=True, use_rmsnorm=True,
            wo_shift=False,
            in_context_len=self.in_context_len,
            in_context_start=self.in_context_start,
            max_tokens=max_tokens,
            block_norm=True,
        )

        self.encoder_layer_norm = nn.LayerNorm(self.diffusion_input_dim, elementwise_affine=True)

        latent_tokens_init = get_1d_sincos_pos_embed_from_grid(encoder_hidden_size, np.arange(self.num_latent_tokens, dtype=np.float64))
        self.latent_tokens = nn.Parameter(torch.from_numpy(latent_tokens_init).unsqueeze(0).to(dtype=torch.float32), requires_grad=True)



    def _setup_timestep_shift(self, cfg):
        alpha_manual = float(self.cfg.get("timestep_shift_alpha", 1.0))
        base_dim = float(self.cfg.get("timestep_shift_base_dim", 4096))
        eff_mode = self.cfg.get("timestep_shift_effective_dim_mode", "total")

        if self.timestep_shift_mode == "disabled":
            self.timestep_shift_alpha = 0.0
        elif self.timestep_shift_mode == "manual":
            self.timestep_shift_alpha = alpha_manual
        elif self.timestep_shift_mode == "auto":
            if eff_mode == "total":
                eff_dim = self.num_latent_tokens * self.diffusion_input_dim
            elif eff_mode == "per_token":
                eff_dim = self.diffusion_input_dim
            elif eff_mode == "num_tokens":
                eff_dim = self.num_latent_tokens
            else:
                raise ValueError(f"Unknown timestep_shift_effective_dim_mode: {eff_mode}")
            self.timestep_shift_alpha = math.sqrt(eff_dim / base_dim)
        else:
            raise ValueError(f"Unknown timestep_shift_mode: {self.timestep_shift_mode}")



    def _build_decoder(self):
        decoder_model_size = self.cfg.get("decoder_model_size", "Base")
        assert decoder_model_size in DECODER_CONFIGS, f"Unknown decoder: {decoder_model_size}"
        dec_layers, dec_hidden, dec_heads, dec_mlp = DECODER_CONFIGS[decoder_model_size]

        decoder_config = DecoderConfig(
            hidden_size=768,  # Decoder input is always 768 (from up_sample_decoder)
            patch_size=self.patch_size,
            decoder_hidden_size=dec_hidden,
            decoder_num_hidden_layers=dec_layers,
            decoder_num_attention_heads=dec_heads,
            decoder_intermediate_size=dec_mlp,
        )
        self.decoder = Decoder(decoder_config, num_patches=self.num_image_patches)
        self.up_sample_decoder = nn.Linear(self.diffusion_input_dim, 768, bias=True)


    
    def _build_denoiser(self, cfg):
        self.transport = Transport(
            train_eps=5e-2, sample_eps=5e-2,
            use_cosine_loss=True, use_lognorm=True,
            lognorm_mu=float(self.cfg.get("lognorm_mu", 0.0)),
            lognorm_sigma=float(self.cfg.get("lognorm_sigma", 1.0)),
            partitial_train=None,
            partial_ratio=1.0,
            shift_lg=False,
        )
        self.flow_sampler = Sampler(self.transport)
        self.flow_steps_per_recon = int(cfg.get("flow_steps_per_recon", 1))
        self.flow_mini_batch = max(1, int(cfg.get("flow_mini_batch", 4)))
        self.gradient_checkpointing_denoiser = bool(cfg.get("gradient_checkpointing_denoiser", self.flow_steps_per_recon > 4))

    def get_learnable_params(self, muon=False, non_muon=False):
        params = []
        
        # Muon params: 2D+ weights from encoder/decoder (if enabled), excluding final_layer / in_context_posemb / latent_tokens (v3 behavior).
        # AdamW params: everything not selected by Muon, plus special params and <2D params.      
        for name, param in self.named_parameters():

            if muon:
                if "patch_embed" in name or "final_layer" in name or "in_context_posemb" in name or "latent_tokens" in name or param.dim() < 2: continue

            if non_muon:
                if "patch_embed" not in name and "final_layer" not in name and "in_context_posemb" not in name and "latent_tokens" not in name and param.dim() >= 2: continue
            
            if not param.requires_grad: continue
            params.append(param)
            if not ("decoder" in name or "encoder" in name or "latent_tokens" in name or "patch_embed" in name):
                raise ValueError(
                    f"Unknown parameter '{name}' (dim={param.dim()}) "
                )
        return params

    def encode(self, imgs):
        img_embed = self.patch_embed(imgs)
        z_noise = torch.randn(
            imgs.shape[0], self.num_latent_tokens, self.diffusion_input_dim,
            device=imgs.device, dtype=imgs.dtype,
        )
        
        # Force 100% class dropout during encoding (class label is arbitrary placeholder)
        # y is never used during reconstruction, all masked out.
        y_for_enc = torch.full((imgs.shape[0],), 7, device=imgs.device, dtype=torch.long)
        force_drop_ids = torch.ones(imgs.shape[0], device=imgs.device, dtype=torch.long)
    
        pos_embed = self.latent_tokens[:, :self.num_latent_tokens].expand(imgs.shape[0], -1, -1)
        flow_t = torch.rand(imgs.shape[0], device=imgs.device) * self.modulation_recon_timestep_max
        z = self.encoder(
            z_noise, t=flow_t, y=y_for_enc, pos_embed=pos_embed,
            img_patch_embed=img_embed,
            force_drop_ids_y_embedder=force_drop_ids,
        )
        return z

    def decode(self, z, representation_phase=False):
        if representation_phase and self.training: z = self.noising(z)
        z = self.up_sample_decoder(z)
        logits = self.decoder(z, drop_cls_token=False)
        return self.decoder.unpatchify(logits)


    def noising(self, x, sampling_prob=0.5):
        B = x.shape[0]
        mask = (torch.rand(B, device=x.device) < sampling_prob).view(B, *([1] * (x.dim() - 1)))
        t, x0, x1 = self.transport.sample(x, sp_timesteps=[self.noising_t_start, 1.0])
        _, x_t, _ = self.transport.path_sampler.plan(t, x0, x1)
        return torch.where(mask, x_t, x)


    def forward(self, imgs, labels):
        recon_loss, output_dict = self.forward_tokenizer(imgs)
        total_loss = recon_loss

        z = output_dict["learned_latent_tokens"]
        gen_dict = self.forward_denoising(z, y=labels)
        flow_loss = gen_dict["flow/flow_loss"]
        total_loss = total_loss + self.gen_loss_weight * flow_loss
        output_dict.update(gen_dict)

        output_dict["total_loss"] = total_loss
        return total_loss, output_dict

    def forward_tokenizer(self, imgs):
        z = self.encode(imgs)
        z_normed = self.encoder_layer_norm(z)
        recon = self.decode(z_normed, representation_phase=True)
        rec_loss, loss_dict = self._reconstruction_loss(recon, imgs)
        loss_dict["learned_latent_tokens"] = z_normed
        loss_dict["recon"] = recon
        return rec_loss, loss_dict


    def forward_denoising(self, z, y=None):
        z_for_flow = z.detach()
        t_list = [self.transport.sample(z_for_flow, timestep_shift=self.timestep_shift_alpha)[0] for _ in range(self.flow_steps_per_recon)]
        flow_loss = None
        
        offset = 0
        for x_chunk in self._iter_flow_step_chunks(self.flow_steps_per_recon, chunk_size=self.flow_mini_batch):
            t_chunk = torch.cat(t_list[offset: offset + x_chunk], dim=0)
            offset += x_chunk

            z_chunk = z_for_flow.repeat(x_chunk, 1, 1)
            y_chunk = y.repeat(x_chunk) if y is not None else None

            pos_embed_chunk = self.latent_tokens[:, :z.shape[1]].expand(z_chunk.shape[0], -1, -1)
            model_kwargs = dict(
                y=y_chunk,
                pos_embed=pos_embed_chunk,
                checkpoint_blocks=self.gradient_checkpointing_denoiser,
            )

            chunk_flow_dict = self.transport.training_losses(
                self.encoder, z_chunk, t_chunk, model_kwargs=model_kwargs
            )
            chunk_flow_loss = self._compute_flow_loss(chunk_flow_dict)
            weighted_chunk_loss = chunk_flow_loss * x_chunk
            flow_loss = weighted_chunk_loss if flow_loss is None else (flow_loss + weighted_chunk_loss)
        
        output = {"flow/flow_loss": flow_loss,}
        return output

    @staticmethod
    def _iter_flow_step_chunks(total_steps: int, chunk_size: int = 4):
        """Yield contiguous flow-step chunk sizes (e.g., 14 -> 4,4,4,2)."""
        if total_steps <= 0:
            return []
        full_chunks, remainder = divmod(total_steps, chunk_size)
        chunks = [chunk_size] * full_chunks
        if remainder > 0:
            chunks.append(remainder)
        return chunks
    

    def _reconstruction_loss(self, recon, imgs):
        rec_loss = F.l1_loss(recon, imgs)
        lpips_loss = self.lpips_loss(imgs, recon)
        total = rec_loss + self.perceptual_weight * lpips_loss
        return total, {"rec_loss": rec_loss, "lpips_loss": lpips_loss, "recon_total": total}

    def _compute_flow_loss(self, flow_dict):
        x1 = flow_dict["x1"]
        xt = flow_dict["xt"]
        t = flow_dict["sampled_t"][:, None, None]
        flow_dict["model_output"] = self.encoder_layer_norm(flow_dict["model_output"])
        
        x_pred = flow_dict["model_output"]
        v_gt = (x1 - xt) / (1 - t).clamp_min(self.transport.train_eps)
        v_pred = (x_pred - xt) / (1 - t).clamp_min(self.transport.train_eps)

        loss = (v_gt - v_pred) ** 2
        loss = loss.mean(dim=(1, 2))
        return loss.mean()



    @torch.no_grad()
    def reconstruct_batch(self, imgs, labels=None):
        z = self.encode(imgs)
        z = self.encoder_layer_norm(z)
        return self.decode(z, representation_phase=False).clamp(0, 1)


    @torch.no_grad()
    def diffusion_generate(self, device, num_visuals=16, cfg_scale=4.0,
                           cfg_interval=(0.0, 1.0), cfg_norm_order=None,
                           y_given=None, noise=None, 
                           return_y=False, return_z=False):
        """Generate images via ODE sampling with classifier-free guidance."""
        
        
        # Labels
        if y_given is None:
            y = torch.tensor([i % self.num_classes for i in range(num_visuals)], device=device)
        else:
            y = y_given

        # Noise
        if noise is None:
            noise = torch.randn(num_visuals, self.num_latent_tokens, self.diffusion_input_dim, device=device)

        # Double batch for CFG
        if cfg_scale > 1.0:
            y_null = torch.full((num_visuals,), self.num_classes, device=device, dtype=torch.long)
            y = torch.cat([y, y_null], 0)
            noise = torch.cat([noise, noise], 0)

        # Pos embed
        pos_embed = self.latent_tokens[:, :self.num_latent_tokens].expand(noise.shape[0], -1, -1)

        # ODE sampler
        sample_fn = self.flow_sampler.sample_ode(
            num_steps=50, timestep_shift=self.timestep_shift_alpha,
        )

        effective_order = cfg_norm_order or self.cfg_norm_order

        def _model_fn(x_t, t, **_unused):
            t_b = t.view(-1, 1, 1)
            denom = (1.0 - t_b).clamp_min(self.transport.sample_eps)

            # We are always normalizing first before CFG. Normazling after CFG or both before/after doesn't change result much.
            if cfg_scale > 1.0:
                x_pred = self.encoder.forward_with_cfg(
                    x_t, t, y=y, pos_embed=pos_embed, cfg_scale=cfg_scale,
                    cfg_interval=cfg_interval,
                    bn_func=self.encoder_layer_norm,
                    cfg_norm_order=effective_order,
                )
            else:
                x_pred = self.encoder.forward(
                    x_t, t, y=y, pos_embed=pos_embed,
                )
                if effective_order in ("cfg_first", "both"):
                    x_pred = self.encoder_layer_norm(x_pred)
            return (x_pred - x_t) / denom
            

        samples = sample_fn(noise, _model_fn)[-1]

        if cfg_scale > 1.0:
            samples, _ = samples.chunk(2, dim=0)
            y_ret = y[:num_visuals]
        else:
            y_ret = y

        imgs = self.decode(samples, representation_phase=False)

        if return_z:
            return imgs, y_ret.cpu().numpy(), samples
        if return_y:
            return imgs, y_ret.cpu().numpy()
        return imgs