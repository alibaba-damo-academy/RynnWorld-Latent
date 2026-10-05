"""DA3 feature representations for representation-probing experiments.

These complement the deliverable latent (ktoken / ktoken_zcam) with the raw
finetuned-encoder patch field, to measure how much action-relevant information
the compressed bottleneck keeps:

  encoder_pooled   mean over patches, per frame        -> [B, 2, feature_dim]
  encoder_diff     pooled(target) - pooled(source)     -> [B, 1, feature_dim]
  encoder_patches  full patch field of both frames     -> [B, 2N, feature_dim]

encoder_patches preserves the spatial layout; feed it to a learned-query
pooling probe head rather than flattening.
"""

from contextlib import nullcontext

import torch

DA3_REPRESENTATIONS = ("encoder_pooled", "encoder_diff", "encoder_patches")


@torch.inference_mode()
def da3_tokens(encoder, images, representation):
    """Compute a DA3-feature representation for RGB frame pairs.

    encoder: a rynnlam.inference.RynnLAMEncoder instance.
    images:  [B, 2, H, W, 3] float RGB in [0, 1].
    """
    if representation not in DA3_REPRESENTATIONS:
        raise ValueError(
            f"Unknown DA3 representation {representation!r}; "
            f"expected one of {DA3_REPRESENTATIONS}"
        )
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if encoder.device.type == "cuda" and encoder.precision == "bf16"
        else nullcontext()
    )
    with context:
        features = encoder.model.encode_pair(images, normalize=encoder.normalize)
    source = features["source_features"]
    target = features["target_features"]
    if representation == "encoder_patches":
        tokens = torch.cat([source, target], dim=1)
    else:
        pooled = torch.stack([source.mean(dim=1), target.mean(dim=1)], dim=1)
        if representation == "encoder_diff":
            tokens = pooled[:, 1:] - pooled[:, :-1]
        else:
            tokens = pooled
    return tokens.float()
