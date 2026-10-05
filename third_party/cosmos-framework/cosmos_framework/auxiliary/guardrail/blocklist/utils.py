# ---------------------------------------------------------------------------
# Provenance reference (composition scan; RynnWorld-LA report 2026-09).
# Scanner flagged 92% snippet similarity to: GEN3C <https://github.com/nv-tlabs/GEN3C>
# That is the nearest public-repo match -- frequently a downstream reuser of the
# same upstream code, NOT a verified derivation. This file's own license, per
# its header, is: OpenMDW-1.1. Shipped unchanged; no relicense is implied. See
# THIRD_PARTY_LICENSES.md -> "Composition-scan review".
# ---------------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
import re

from cosmos_framework.utils import log


def read_keyword_list_from_dir(folder_path: str) -> list[str]:
    """Read keyword list from all files in a folder."""
    output_list = []
    file_list = []
    # Get list of files in the folder
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_list.append(file)

    # Process each file
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        try:
            with open(file_path) as f:
                output_list.extend([line.strip() for line in f.readlines()])
        except Exception as e:
            log.error(f"Error reading file {file}: {e!s}")

    return output_list


def to_ascii(prompt: str) -> str:
    """Convert prompt to ASCII."""
    return re.sub(r"[^\x00-\x7F]+", " ", prompt)
