"""Shared RynnWorld-Latent rollout dataset, model-loading, batch, and metric helpers."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent.parent


def resolve_arg_paths(args, *names: str) -> None:
    """Expand ``~`` and resolve each named argparse attribute to an absolute path.

    Empty or missing attributes are left untouched, so optional flags stay
    optional. Mutates ``args`` in place.
    """
    for name in names:
        value = getattr(args, name, None)
        if value:
            setattr(args, name, str(Path(value).expanduser().resolve()))


def _resolve_rollout_paths(args):
    """Resolve the rollout flags, then publish the two the experiment config reads.

    ``MANIFEST_DIR`` / ``STAGED_ROOT`` are pulled in via ``${oc.env:...}`` while the
    experiment config is instantiated, so they must be in the environment before the
    model is built.
    """
    resolve_arg_paths(args, "checkpoint", "manifest_dir", "staged_root", "out")
    for name, env_name in (("manifest_dir", "MANIFEST_DIR"), ("staged_root", "STAGED_ROOT")):
        value = getattr(args, name, None)
        if value is not None:
            os.environ[env_name] = str(value)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR between two [0,1] tensors of identical shape."""
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def crop_common(a: torch.Tensor, b: torch.Tensor):
    """Center-crop two [B,C,T,H,W] tensors to their common T/H/W so metrics are comparable."""
    t = min(a.shape[2], b.shape[2])
    h = min(a.shape[3], b.shape[3])
    w = min(a.shape[4], b.shape[4])

    def _c(x):
        dh = (x.shape[3] - h) // 2
        dw = (x.shape[4] - w) // 2
        return x[:, :, :t, dh : dh + h, dw : dw + w]

    return _c(a), _c(b)


def temporal_diff(x: torch.Tensor) -> float:
    """Mean |x[t+1]-x[t]| over time — a proxy for how much motion a clip contains."""
    if x.shape[2] < 2:
        return 0.0
    return torch.mean((x[:, :, 1:] - x[:, :, :-1]).abs()).item()


def pm1_to_unit(x: torch.Tensor) -> torch.Tensor:
    """Model output in [-1,1] -> [0,1], the range the metrics and video savers expect."""
    return (x.detach().float().clamp(-1, 1) + 1) / 2


def build_rollout_dataset(args):
    from rynnworld_latent.manifest_dataset import RynnWorldManifestDataset

    _resolve_rollout_paths(args)
    return RynnWorldManifestDataset(
        manifest_dir=args.manifest_dir,
        staged_root=args.staged_root or None,
        # The released weights are a v3 (fps=None) checkpoint trained on each
        # record's own probed fps (the bundled samples are 30 fps). fps is the
        # mRoPE temporal-step denominator (base_fps/fps = 24/fps), so eval uses
        # fps=None to match the training-time temporal encoding exactly.
        fps=None,
        mode="forward_dynamics",
        action_normalization="quantile",
        viewpoint="ego_view",
        limit=0,
        max_retries=8,
    )


def build_rollout_batch(model, sample: dict, device):
    """Hand-build a forward_dynamics inference batch from a dataset sample.

    Uses our real (video, action) pair together with the trainer's own
    sequence-plan units, so inference matches the training contract.
    """
    from cosmos_framework.data.generator.action.transforms import (
        build_sequence_plan_from_mode,
        find_closest_target_size,
        reflection_pad_to_target,
    )
    from cosmos_framework.data.generator.action.action_processing import (
        ActionProcessingRecord,
        make_batched_action_processing_fields,
    )
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption

    video_cthw = sample["video"]  # [C,T,H,W] uint8, T=81
    action = sample["action"].to(torch.float32)  # [20,608], quantile-normalized
    caption = sample["ai_caption"]
    if os.environ.get("RYNNWORLD_TEXT_FREE", "0").strip() == "1":
        caption = ""  # text-free checkpoint: condition only on first frame + action
    fps = int(sample["conditioning_fps"].item())
    C, T, H, W = video_cthw.shape

    # Aspect-preserving resize + reflection-pad to the resolution-tier bucket, exactly
    # as training does. Raw decode (e.g. 480x854) has a width not divisible by the VAE
    # 16x factor, which makes the sampler's latent-grid reshape fail; the 16:9 bucket
    # (832x480) is divisible by 16.
    v = video_cthw.to(torch.float32) / 127.5 - 1.0  # [C,T,H,W] in [-1,1]
    target_w, target_h = find_closest_target_size(H, W, "480")
    pad_dict = {"video": v}
    reflection_pad_to_target(
        pad_dict, ["video"], keep_aspect_ratio=True, target_w=target_w, target_h=target_h
    )
    v = pad_dict["video"]  # [C,T,target_h,target_w]
    v = v.clamp(-1.0, 1.0)  # resize interpolation can overshoot; model asserts [-1,1]
    image_size = pad_dict["image_size"].to(torch.float32).view(1, 4)
    video = v.unsqueeze(0).to(device)  # [1,C,T,target_h,target_w]
    action = action.to(device)
    raw_action_dim = action.shape[-1]

    # video_length is PIXEL frames (transforms.py:682 uses video.shape[1]); Case C.
    sequence_plan = build_sequence_plan_from_mode(
        mode="forward_dynamics",
        video_length=T,
        action_length=action.shape[0],
        video_temporal_downsample=4,
    )

    ids = tokenize_caption(
        caption, model.vlm_tokenizer, is_video=False,
        use_system_prompt=model.vlm_config.use_system_prompt,
    )
    text_token_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    batch = {
        model.input_video_key: [video],
        "action": [action],
        "mode": ["forward_dynamics"],
        model.input_caption_key: [caption],
        "text_token_ids": [text_token_ids],
        "image_size": [image_size.to(device)],
        "fps": torch.tensor([float(fps)], device=device),
        "conditioning_fps": torch.tensor([float(fps)], device=device),
        "num_frames": torch.tensor([T], device=device),
        "domain_id": [sample["domain_id"].to(device)],
        "sequence_plan": [sequence_plan],
        "is_preprocessed": True,
        **make_batched_action_processing_fields(
            ActionProcessingRecord(raw_action_dim=raw_action_dim, action_normalizer=None), 1
        ),
    }
    # The decode returns the unpadded content region floored to a VAE multiple of 16,
    # while `video` is the reflection-padded canvas (mirror pad at bottom+right).
    # Return the matching content crop as GT, else PSNR compares gen against mirror
    # borders and can score below the static-first-frame baseline.
    oh, ow = int(image_size[0, 2].item()), int(image_size[0, 3].item())
    gt = video[:, :, :, :oh, :ow]
    gt = gt[:, :, :, : gt.shape[3] // 16 * 16, : gt.shape[4] // 16 * 16]
    return batch, gt


def _patch_processor_local():
    """Redirect the Cosmos3-Edge VLM tokenizer to local files (no HF Hub download).

    Offline inference environments have no reliable HF access; build_processor_lazy
    would otherwise shell out to ``uvx hf download nvidia/Cosmos3-Edge`` and hang.
    Mirrors scripts/train.py:_patch_processor_local.
    """
    edge_dir = os.environ.get("RYNNWORLD_EDGE_HF_DIR", "") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party", "cosmos_tokenizers", "edge",
    )
    tok_dir = os.path.join(edge_dir, "text_tokenizer")
    tok_file = os.path.join(tok_dir, "tokenizer.json")
    if not os.path.isfile(tok_file):
        print(f"[eval] WARNING: local tokenizer missing at {tok_file}; processor patch skipped")
        return
    import types

    import cosmos_framework.data.generator.processors as _proc
    import cosmos_framework.utils.checkpoint_db as _ckptdb
    from transformers import PreTrainedTokenizerFast

    class _FakeProcessor:
        def __init__(self):
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tok_file, bos_token="<s>", eos_token="</s>", pad_token="<pad>"
            )
            chat_tpl = os.path.join(tok_dir, "chat_template.jinja")
            if os.path.isfile(chat_tpl):
                with open(chat_tpl) as fh:
                    self.tokenizer.chat_template = fh.read()
            self.processor = types.SimpleNamespace(tokenizer=self.tokenizer)

        @property
        def eos_id(self):
            return self.tokenizer.eos_token_id

        def tokenize_text(self, caption, is_video=False, use_system_prompt=False, system_prompt=None):
            from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption

            return tokenize_caption(
                caption, self.tokenizer, is_video=is_video,
                use_system_prompt=use_system_prompt, system_prompt=system_prompt,
            )

        def encode(self, *a, **k):
            return self.tokenizer.encode(*a, **k)

        def decode(self, *a, **k):
            return self.tokenizer.decode(*a, **k)

    _orig = _ckptdb.CheckpointDirHf._download

    def _local_download(self):
        if getattr(self, "repository", "") == "nvidia/Cosmos3-Edge":
            return edge_dir
        return _orig(self)

    _ckptdb.CheckpointDirHf._download = _local_download
    _proc.build_processor = lambda local_path=None, *a, **k: _FakeProcessor()
    print(f"[eval] processor patched -> local tokenizer: {tok_file}")


def load_world_model(args):
    _resolve_rollout_paths(args)

    # Import our experiment so the experiment name registers in the ConfigStore.
    import rynnworld_latent.experiment_config  # noqa: F401

    _patch_processor_local()

    # Machines without flash_attn need the varlen SDPA fallback registered before
    # any attention runs; training does this in scripts/train.py, inference here,
    # both via the same shared implementation.
    from rynnworld_latent.attention_sdpa_fallback import patch_sdpa_attention_backend

    patch_sdpa_attention_backend()

    # Rebuild the same action-conditioning structure used in training (two-tower
    # encoder, frame injection + FiLM; opt-in via the RYNNWORLD_ACTION_* env vars) so
    # the checkpoint's action params load, and make classifier-free guidance act on
    # the ACTION (uncond branch zeroes it).
    from rynnworld_latent.action_conditioning import (
        register_action_conditioning_patches,
        patch_inference_action_guidance,
    )

    register_action_conditioning_patches()
    patch_inference_action_guidance()

    # The experiment config's tokenizer vae_path is relative
    # ("pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"); it only resolves from
    # the cosmos-framework root, so chdir there before building the model.
    cosmos_root = os.environ.get(
        "COSMOS_ROOT",
        str(_HERE / "third_party" / "cosmos-framework"),
    )
    if cosmos_root and os.path.isdir(cosmos_root):
        os.chdir(cosmos_root)

    from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint

    # keys_to_skip_loading=[] is CRITICAL: the training config skips loading
    # action2llm/llm2action/action_modality_embed (to preserve the reseed init when
    # loading the BASE checkpoint). When loading OUR fine-tuned checkpoint we must
    # load those trained weights, so override the skip list to empty.
    model, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint,
        config_file="cosmos_framework/configs/base/config.py",
        load_ema_to_reg=not args.no_ema,
        keys_to_skip_loading=[],
        # Disable torch.compile for inference. The experiment config defaults
        # compile.enabled=True, but the training TOML pins it False; compiling the
        # self-attn region can raise torch._dynamo.exc.Unsupported on some torch
        # builds (suppress_errors does not catch this "Observed exception"). Match
        # training: compile off -> eager.
        compile_config={"enabled": False},
        seed=args.seed,
    )
    model.eval()
    return model
