#!/usr/bin/env python3
"""Convert released RynnWorld-Latent safetensors (``net.*`` layout) -> DCP for warm-start.

The published HF / ModelScope weights are safetensors whose keys are ``net.*``
(regular) plus ``net_ema.*`` (EMA mirrors) — i.e. exactly the ``OmniMoTModel``
state-dict layout a cosmos DCP training checkpoint stores under ``<iter>/model/``.
The cosmos trainer warm-starts (``checkpoint.load_path``) only from **DCP**, not
safetensors, so downstream post-training from the released weights needs this
one-time re-serialization.

It reads the shards and re-saves the tensors with cosmos's ``CustomSavePlanner``
— **no model instantiation, CPU-only, no GPU and no Wan2.2 VAE required** (unlike
``convert_edge_to_dcp.py``, which builds the base Edge model). Output layout matches
a real training checkpoint: ``<out>/model/.metadata`` + ``<out>/model/*.distcp``.
Point ``BASE_CHECKPOINT_PATH`` (``checkpoint.load_path``) at ``<out>``.

The released model is a film checkpoint, so post-training warm-starts with
``keys_to_skip_loading=["net_ema."]`` (the ``rynnworld_latent_edge_posttrain``
recipe): the ``net.*`` weights load, ``net_ema.*`` is ignored and re-warms from
them. Pass ``--skip-ema`` to drop ``net_ema.*`` and halve the output size.

Usage::

    python scripts/checkpoints/convert_released_to_dcp.py \
        --safetensors /path/to/RynnWorld-Latent \
        --out         /path/to/RynnWorld-Latent-dcp
    # then:  export BASE_CHECKPOINT_PATH=/path/to/RynnWorld-Latent-dcp
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COSMOS_ROOT = Path(os.environ.get("COSMOS_ROOT", str(_REPO_ROOT / "third_party" / "cosmos-framework")))
sys.path.insert(0, str(_COSMOS_ROOT))
sys.path.insert(0, str(_REPO_ROOT))


def _patch_save_planner_compat() -> None:
    """cosmos ``CustomSavePlanner`` forwards ``enable_plan_caching`` to torch's
    ``DefaultSavePlanner``, which torch<2.7 does not accept -> ``TypeError``. This
    mirrors ``scripts/train.py``'s shim so the converter runs on the same torch
    range training does. No-op where torch already supports plan caching (>=2.7).
    """
    import inspect

    import torch.distributed.checkpoint.default_planner as _dp
    from cosmos_framework.checkpoint import dcp as _cdcp

    if "enable_plan_caching" in inspect.signature(_dp.DefaultSavePlanner.__init__).parameters:
        return

    _Base = _dp.DefaultSavePlanner

    def _compat_init(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        dedup_save_to_lowest_rank: bool = False,
        save_reg_to_ema: bool = False,
        enable_plan_caching: bool = False,  # accepted but ignored on torch<2.7
        cache_plans_key=None,
    ) -> None:
        _Base.__init__(
            self,
            flatten_state_dict=flatten_state_dict,
            flatten_sharded_tensors=flatten_sharded_tensors,
            dedup_save_to_lowest_rank=dedup_save_to_lowest_rank,
        )
        if cache_plans_key is not None:
            self._cached_plans_key = cache_plans_key
        self.save_reg_to_ema = save_reg_to_ema

    _cdcp.CustomSavePlanner.__init__ = _compat_init
    print("[convert-released] patched CustomSavePlanner.__init__ (dropped enable_plan_caching for torch<2.7)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safetensors", required=True, help="Released RynnWorld-Latent safetensors dir (net.* layout)")
    ap.add_argument("--out", required=True, help="Output DCP dir; weights go to <out>/model/. Use as BASE_CHECKPOINT_PATH.")
    ap.add_argument("--skip-ema", action="store_true", help="drop net_ema.* (post-train skips it anyway; halves the output)")
    args = ap.parse_args()

    src = Path(args.safetensors).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"safetensors dir not found: {src}")
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no *.safetensors under {src} (is this the released weights dir?)")
    out_model = out / "model"
    if out_model.exists():
        raise SystemExit(f"refusing to overwrite existing {out_model}; choose a new --out")

    from safetensors.torch import load_file

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.filesystem import FileSystemWriter

    from cosmos_framework.checkpoint.dcp import CustomSavePlanner

    _patch_save_planner_compat()  # torch<2.7 CustomSavePlanner shim (no-op on >=2.7)

    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        for key, tensor in load_file(str(shard)).items():
            if args.skip_ema and key.startswith("net_ema."):
                continue
            if key in state_dict:
                raise SystemExit(f"duplicate key across shards: {key}")
            state_dict[key] = tensor

    n_net = sum(1 for k in state_dict if k.startswith("net.") and not k.startswith("net_ema."))
    n_ema = sum(1 for k in state_dict if k.startswith("net_ema."))
    print(f"[convert-released] read {len(state_dict)} tensors from {len(shards)} shard(s): {n_net} net.*, {n_ema} net_ema.*")
    if n_net == 0:
        raise SystemExit(
            "no net.* keys found — this does not look like a RynnWorld-Latent release "
            "checkpoint (expected the raw OmniMoTModel net.* / net_ema.* layout)."
        )

    nbytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    # Match transformers' 5GB shard size so the writer parallelizes like the base converter.
    thread_count = max(1, math.ceil(nbytes / (5 * 1024**3)))
    out_model.mkdir(parents=True, exist_ok=True)

    print(f"[convert-released] saving DCP ({nbytes / 1e9:.2f} GB, thread_count={thread_count}) -> {out_model}")
    dcp.save(
        state_dict=state_dict,
        storage_writer=FileSystemWriter(str(out_model), thread_count=thread_count),
        planner=CustomSavePlanner(),
    )
    print(f"[convert-released] done. DCP checkpoint at: {out}")
    print(f"[convert-released]   export BASE_CHECKPOINT_PATH={out}")


if __name__ == "__main__":
    main()
