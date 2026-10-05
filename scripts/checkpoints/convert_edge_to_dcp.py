#!/usr/bin/env python3
"""Convert Cosmos3-Edge (HuggingFace/safetensors) -> DCP for cosmos SFT training.

Why not the stock ``cosmos_framework.scripts.convert_model_to_dcp``?
It builds the model config from the checkpoint's root ``config.json``, which for
Cosmos3-Edge only stores a *partial* ``model.config`` (vlm_config +
diffusion_expert_config). ``Cosmos3OmniModel.__init__`` then crashes with
``Missing key ema``.

This script builds the COMPLETE model config from the Cosmos3-Edge inference YAML
(the same proven path the rollout/inference code uses), points the VAE + reasoner
backbone at the local checkpoint, loads the HF weights via
``Cosmos3OmniModel.from_pretrained_dcp`` (handles the diffusers layout), and saves
the full model state dict with cosmos's ``CustomSavePlanner``.

Output layout matches what the trainer's DistributedCheckpointer expects for a
warm-start base checkpoint: ``<out>/model/*.distcp`` + ``<out>/model/.metadata``
(+ ``config.json``). Point ``checkpoint.load_path`` at ``<out>``.

Usage::

    COSMOS_DEVICE=cpu python scripts/checkpoints/convert_edge_to_dcp.py \
        --hf-src /path/Cosmos3-Edge --vae /path/Wan2.2_VAE.pth --out /path/Cosmos3-Edge-dcp
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_COSMOS = _REPO_ROOT / "third_party" / "cosmos-framework"
COSMOS_ROOT = Path(os.environ.get("COSMOS_ROOT", str(_DEFAULT_COSMOS)))
sys.path.insert(0, str(COSMOS_ROOT))
sys.path.insert(0, str(_REPO_ROOT))
EDGE_YAML = COSMOS_ROOT / "cosmos_framework/inference/configs/model/Cosmos3-Edge.yaml"

if TYPE_CHECKING:
    from cosmos_framework.inference.model import Cosmos3OmniConfig

_CONFIG_REPLACEMENTS = [
    ("cosmos3._src.vfm.configs.base.defaults.vlm.", "cosmos_framework.configs.base.defaults.reasoner."),
    ("cosmos3._src.vfm.configs.base.", "cosmos_framework.configs.base."),
    ("cosmos3._src.vfm.models.", "cosmos_framework.model.generator."),
    ("cosmos3._src.vfm.tokenizers.", "cosmos_framework.model.generator.tokenizers."),
    ("cosmos3._src.imaginaire.", "cosmos_framework."),
    ("cosmos3/_src/vfm/models/vlm/", "cosmos_framework/model/generator/reasoner/"),
    ("cosmos3/_src/vfm/", "cosmos_framework/"),
]


def _patch_attention_sdpa():
    """Use PyTorch native SDPA (no flash-attn dependency) during instantiation."""
    import torch.nn.functional as F
    import cosmos_framework.model.attention.frontend as _attn_frontend

    def _sdpa_attention(query, key, value, is_causal=False, scale=None, **kwargs):
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        nq, nk = q.shape[1], k.shape[1]
        if nq != nk:
            rep = nq // nk
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
        return out.transpose(1, 2)

    _attn_frontend.attention = _sdpa_attention
    import cosmos_framework.model.attention as _attn_mod

    if hasattr(_attn_mod, "attention"):
        _attn_mod.attention = _sdpa_attention


def _patch_processor_local(hf_src: Path):
    """Build the VLM processor from the local tokenizer files (no HF download)."""
    import cosmos_framework.data.generator.processors as _proc_mod
    from transformers import PreTrainedTokenizerFast

    tok_file = hf_src / "text_tokenizer" / "tokenizer.json"
    chat_tpl_file = hf_src / "text_tokenizer" / "chat_template.jinja"

    class _FakeProcessor:
        def __init__(self):
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tok_file), bos_token="<s>", eos_token="</s>", pad_token="<pad>"
            )
            if chat_tpl_file.exists():
                self.tokenizer.chat_template = chat_tpl_file.read_text()

    def _patched_build_processor(*args, **kwargs):
        return _FakeProcessor()

    _proc_mod.build_processor = _patched_build_processor
    _proc_mod.build_processor_lazy = _patched_build_processor


def build_conversion_config(hf_src: Path, vae_path: Path) -> Cosmos3OmniConfig:
    import yaml
    from cosmos_framework.inference.model import Cosmos3OmniConfig

    raw = EDGE_YAML.read_text()
    for old, new in _CONFIG_REPLACEMENTS:
        raw = raw.replace(old, new)
    cfg = yaml.safe_load(raw)
    model_cfg = cfg["model"]

    mc = model_cfg["config"]
    mc["ema"]["enabled"] = False
    mc["activation_checkpointing"]["mode"] = "none"
    mc["compile"]["enabled"] = False

    # Load the VAE from the local file; disable the object-store backend so it
    # does not try to read credentials/gcp_training.secret.
    mc["tokenizer"]["vae_path"] = str(vae_path)
    mc["tokenizer"]["bucket_name"] = ""
    mc["tokenizer"]["object_store_credential_path_pretrained"] = ""

    # Load the reasoner (Nemotron-2B) backbone from the local HF dir (root
    # model.safetensors) instead of the inaccessible s3 path.
    mc["vlm_config"]["pretrained_weights"]["enabled"] = True
    mc["vlm_config"]["pretrained_weights"]["backbone_path"] = str(hf_src)
    mc["vlm_config"]["pretrained_weights"]["credentials_path"] = ""
    mc["vlm_config"]["pretrained_weights"]["enable_gcs_patch_in_boto3"] = False
    # Diffusion net weights are loaded by from_pretrained_dcp's diffusers path.
    mc["diffusion_expert_config"]["load_weights_from_pretrained"] = False

    return Cosmos3OmniConfig(model=model_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-src", required=True, help="Local Cosmos3-Edge HF/safetensors dir")
    ap.add_argument("--vae", required=True, help="Path to Wan2.2_VAE.pth")
    ap.add_argument("--out", required=True, help="Output DCP directory (parent of model/)")
    args = ap.parse_args()

    hf_src = Path(args.hf_src).expanduser().absolute()
    vae_path = Path(args.vae).expanduser().absolute()
    out = Path(args.out).expanduser().absolute()
    if not hf_src.is_dir():
        raise SystemExit(f"HF source not found: {hf_src}")
    if not vae_path.is_file():
        raise SystemExit(f"VAE not found: {vae_path}")

    out_model = out / "model"
    if out_model.exists():
        raise SystemExit(f"Refusing to replace existing checkpoint: {out_model}; choose a new --out directory")
    if not COSMOS_ROOT.is_dir():
        raise SystemExit("Set COSMOS_ROOT to your cosmos-framework checkout (or use the vendored third_party copy).")

    # Parse --help and validate paths before importing or initializing the model stack.
    from cosmos_framework.inference.common.init import init_script

    init_script(env={"COSMOS_DEVICE": "cpu"})

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.filesystem import FileSystemWriter
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    from cosmos_framework.checkpoint.dcp import CustomSavePlanner
    from cosmos_framework.configs.base.defaults.compile import CompileConfig
    from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
    from cosmos_framework.inference.model import Cosmos3OmniModel, _ROOT_DIR

    print(f"[convert] hf_src={hf_src}")
    print(f"[convert] vae={vae_path}")
    print(f"[convert] out={out}")

    _patch_attention_sdpa()
    _patch_processor_local(hf_src)

    config = build_conversion_config(hf_src, vae_path)

    print("[convert] Instantiating model + loading HF weights ...")
    with contextlib.chdir(_ROOT_DIR):
        hf_model = Cosmos3OmniModel.from_pretrained_dcp(
            hf_src,
            config=config,
            parallelism_config=ParallelismConfig(),
            compile_config=CompileConfig(enabled=False),
        )

    state_dict = get_model_state_dict(hf_model.model)

    # Match transformers default max shard size = 5GB.
    max_shard_size = 5 * 1024**3
    model_size = sum(
        p.numel() * p.element_size() for p in state_dict.values() if isinstance(p, torch.Tensor)
    )
    thread_count = max(1, math.ceil(model_size / max_shard_size))
    print(f"[convert] model_size={model_size / 1e9:.2f} GB, thread_count={thread_count}")

    out.mkdir(parents=True, exist_ok=True)

    print("[convert] Saving DCP ...")
    storage_writer = FileSystemWriter(str(out_model), thread_count=thread_count)
    dcp.save(state_dict=state_dict, storage_writer=storage_writer, planner=CustomSavePlanner())
    config.save_pretrained(str(out_model))

    print(f"[convert] Done. DCP checkpoint at: {out}")
    print(f"[convert]   model dir: {out_model}")


if __name__ == "__main__":
    main()
