# ---------------------------------------------------------------------------
# Provenance reference (composition scan; RynnWorld-LA report 2026-09).
# Scanner flagged 89% snippet similarity to: GEN3C <https://github.com/nv-tlabs/GEN3C>
# That is the nearest public-repo match -- frequently a downstream reuser of the
# same upstream code, NOT a verified derivation. This file's own license, per
# its header, is: OpenMDW-1.1. Shipped unchanged; no relicense is implied. See
# THIRD_PARTY_LICENSES.md -> "Composition-scan review".
# ---------------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import cv2
import numpy as np


def pixelate_face(face_img: np.ndarray, blocks: int = 5) -> np.ndarray:
    """
    Pixelate a face region by reducing resolution and then upscaling.

    Args:
        face_img: Face region to pixelate
        blocks: Number of blocks to divide the face into (in each dimension)

    Returns:
        Pixelated face region
    """
    h, w = face_img.shape[:2]
    # Shrink the image and scale back up to create pixelation effect
    temp = cv2.resize(face_img, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated
