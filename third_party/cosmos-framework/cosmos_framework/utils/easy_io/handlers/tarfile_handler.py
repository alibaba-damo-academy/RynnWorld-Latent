# ---------------------------------------------------------------------------
# Provenance reference (composition scan; RynnWorld-LA report 2026-09).
# Scanner flagged 90% snippet similarity to: GEN3C <https://github.com/nv-tlabs/GEN3C>
# That is the nearest public-repo match -- frequently a downstream reuser of the
# same upstream code, NOT a verified derivation. This file's own license, per
# its header, is: OpenMDW-1.1. Shipped unchanged; no relicense is implied. See
# THIRD_PARTY_LICENSES.md -> "Composition-scan review".
# ---------------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import tarfile

from cosmos_framework.utils.easy_io.handlers.base import BaseFileHandler


class TarHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, mode="r|*", **kwargs):
        return tarfile.open(fileobj=file, mode=mode, **kwargs)

    def load_from_path(self, filepath, mode="r|*", **kwargs):
        return tarfile.open(filepath, mode=mode, **kwargs)

    def dump_to_fileobj(self, obj, file, mode="w", **kwargs):
        with tarfile.open(fileobj=file, mode=mode) as tar:
            tar.add(obj, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        with tarfile.open(filepath, mode=mode) as tar:
            tar.add(obj, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError
