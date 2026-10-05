# ---------------------------------------------------------------------------
# Provenance reference (composition scan; RynnWorld-LA report 2026-09).
# Scanner flagged 100% snippet similarity to: TheAnimeScripter <https://github.com/NevermindNilas/TheAnimeScripter>
# That is the nearest public-repo match -- frequently a downstream reuser of the
# same upstream code, NOT a verified derivation. This file's own license, per
# its header, is: see file header / project records. Shipped unchanged; no relicense is implied. See
# THIRD_PARTY_LICENSES.md -> "Composition-scan review".
# ---------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from .attention import MemEffAttention
from .block import Block
from .layer_scale import LayerScale
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .rope import PositionGetter, RotaryPositionEmbedding2D
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused

__all__ = [
    Mlp,
    PatchEmbed,
    SwiGLUFFN,
    SwiGLUFFNFused,
    Block,
    # MemEffAttention,
    LayerScale,
    PositionGetter,
    RotaryPositionEmbedding2D,
]
