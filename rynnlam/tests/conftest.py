"""Shared CPU test helpers for RynnLAM's inference / labeling / eval tests.

The trainer and its loss tests are not part of this inference-only bundle, so
the two loss-free helpers the remaining tests rely on live here.
"""

import torch
from torch import nn


class SyntheticEncoder(nn.Module):
    """Deterministic stand-in for the DA3-Large backbone (no weights needed)."""

    def __init__(self, finetune_mode="freeze", **kwargs):
        super().__init__()
        self.scale = nn.Parameter(
            torch.ones(2048), requires_grad=finetune_mode == "full"
        )
        self.last_images = None

    def forward(self, images):
        self.last_images = images
        values = images[:, :, ::14, ::14].mean(-1).flatten(2)
        channels = torch.arange(2048, device=images.device, dtype=images.dtype) / 2048
        features = torch.sin(values.unsqueeze(-1) + channels) * self.scale
        return [(features, None)], None


def tiny_config(source="features", k=2, mode="full"):
    """A minimal RynnLAMConfig kwargs dict that builds a tiny model on CPU."""
    return dict(
        latent_encoder_model_dim=32,
        latent_encoder_num_heads=4,
        latent_encoder_num_blocks=1,
        flow_decoder_model_dim=32,
        flow_decoder_num_heads=4,
        flow_decoder_dec_blocks=1,
        recon_decoder_model_dim=32,
        recon_decoder_num_heads=4,
        recon_decoder_num_blocks=1,
        cam_dec_hidden_dim=16,
        k_token_source=source,
        num_k_tokens=k,
        encoder_finetune_mode=mode,
    )
