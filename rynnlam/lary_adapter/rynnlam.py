"""Optional LARY get_latent_action-compatible bridge; no registry side effects.

Import RynnLAMLaryAdapter explicitly in your external LARY integration. The
standalone extractor does not use this module or require LARY to be installed.
"""

import numpy as np
import torch

from rynnlam.inference import RynnLAMEncoder
from rynnlam.probe_repr import DA3_REPRESENTATIONS, da3_tokens


class RynnLAMLaryAdapter:
    def __init__(
        self,
        checkpoint,
        representation="ktoken",
        device="cuda",
        normalization="imagenet",
        precision="bf16",
        trust_checkpoint=False,
    ):
        if normalization not in ("imagenet", "none"):
            raise ValueError("normalization must be imagenet or none")
        self.representation = representation
        self.encoder = RynnLAMEncoder(
            checkpoint,
            device=device,
            normalize=normalization == "imagenet",
            precision=precision,
            trust_checkpoint=trust_checkpoint,
        )

    @torch.inference_mode()
    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        """LARY [B,3,2,H,W] RGB floats -> per-pair tokens and empty indices."""
        if batch_data.ndim != 5 or batch_data.shape[1:3] != (3, 2):
            raise ValueError("Expected [B,3,2,H,W] float RGB images in [0,1]")
        images = batch_data.permute(0, 2, 3, 4, 1)
        if self.representation in DA3_REPRESENTATIONS:
            out = da3_tokens(self.encoder, images, self.representation)
        else:
            out = self.encoder(images, representation=self.representation)
        tokens = out.float().cpu().numpy()
        return list(tokens), [np.array([], dtype=np.int64) for _ in tokens]
