import collections.abc

import torch
import torch.nn as nn
from .encoder_utils import get_2d_sincos_pos_embed


class ViTMAEPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, embed_dim)` to be consumed by a
    Transformer.

    Args:
        embed_dim: Output embedding dimension. If None, defaults to hidden_size (backward compatible).
                   For patch_size=8, set embed_dim=192 (8*8*3) then project to hidden_size externally.
    """

    def __init__(self, image_size, patch_size, num_channels, hidden_size, embed_dim=None):
        super().__init__()
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.embed_dim = embed_dim if embed_dim is not None else hidden_size
        self.projection = nn.Conv2d(num_channels, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        if not interpolate_pos_encoding and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


class ViTMAEEmbeddings(nn.Module):
    """
        Copied from: https://github.com/huggingface/transformers/blob/7a833d1ccd41673030c85107f65f454c0c3222f5/src/transformers/models/vit_mae/modeling_vit_mae.py#L165
        Changes:
            - no cls token.
            - no random masking.
            - in this line ```sqrt_num_positions = int(num_positions**0.5)`` replaced torch_init to int

        Args:
            embed_dim: Output embedding dimension for patch embeddings. If None, defaults to hidden_size.
                       For patch_size=8, set embed_dim=192 (8*8*3) then project to hidden_size externally.
    """

    def __init__(self, image_size, patch_size, num_channels, hidden_size, embed_dim=None):
        super().__init__()
        self.embed_dim = embed_dim if embed_dim is not None else hidden_size
        self.patch_embeddings = ViTMAEPatchEmbeddings(
            image_size, patch_size, num_channels, hidden_size, embed_dim=self.embed_dim
        )
        self.num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim), requires_grad=False
        )
        self.patch_size = patch_size

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1],
            int(self.patch_embeddings.num_patches**0.5),
            add_cls_token=False,
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # Initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


    # Copied from transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        # class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5) # from torch_init to int
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if interpolate_pos_encoding:
            position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            position_embeddings = self.position_embeddings
        embeddings = embeddings + position_embeddings[:, 1:, :]
        return embeddings
