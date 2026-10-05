# ---------------------------------------------------------------------------
# Provenance reference (composition scan; RynnWorld-LA report 2026-09).
# Scanner flagged 92% snippet similarity to: mmcv <https://github.com/open-mmlab/mmcv>
# That is the nearest public-repo match -- frequently a downstream reuser of the
# same upstream code, NOT a verified derivation. This file's own license, per
# its header, is: OpenMDW-1.1. Shipped unchanged; no relicense is implied. See
# THIRD_PARTY_LICENSES.md -> "Composition-scan review".
# ---------------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from abc import ABCMeta, abstractmethod


class BaseFileHandler(metaclass=ABCMeta):
    # `str_like` is a flag to indicate whether the type of file object is
    # str-like object or bytes-like object. Pickle only processes bytes-like
    # objects but json only processes str-like object. If it is str-like
    # object, `StringIO` will be used to process the buffer.
    str_like = True

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        pass

    def load_from_path(self, filepath, mode="r", **kwargs):
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)
