#!/usr/bin/env python3
"""Roll out a RynnWorld-Latent checkpoint with a CUSTOM first-frame (e.g. inpainted).

Same pipeline as scripts/inference/rollout.py, but frame 0 of the sample video is replaced
with a user-provided image before building the forward_dynamics batch. The 20
latent actions and caption stay untouched, so the generated motion should match
the original GT while the appearance follows the edited image. This is the
"edit the first frame, reuse the source latent actions" data-synthesis mode.

Usage::

    cd RynnWorld-Latent
    export PYTHONPATH="$PWD:$PWD/third_party/cosmos-framework"
    export MANIFEST_DIR=/path/to/manifest          # or the bundled data/manifest
    export STAGED_ROOT=/path/to/staged480          # optional; empty reads sources in place
    # export the same RYNNWORLD_ACTION_* film switches used in training
    python scripts/inference/inpaint_rollout.py \
        --manifest-dir "$MANIFEST_DIR" --staged-root "$STAGED_ROOT" \
        --checkpoint /path/to/checkpoints/iter_000008000 \
        --index 0 \
        --cond-image /path/to/edited_first_frame.png \
        --out outputs/inpaint_rollout.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rynnworld_latent.inference import (  # noqa: E402
    build_rollout_batch,
    build_rollout_dataset,
    load_world_model,
    pm1_to_unit,
    resolve_arg_paths,
)


def main():
    ap = argparse.ArgumentParser(description="Inpainted-first-frame rollout")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--staged-root", default="")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", default="rynnworld_latent_edge_manifest_fullft")
    ap.add_argument("--index", type=int, default=0, help="dataset index of the source sample")
    ap.add_argument("--cond-image", required=True, help="edited first-frame image (png/jpg)")
    ap.add_argument("--out", default="outputs/inpaint_rollout.mp4")
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()
    resolve_arg_paths(args, "manifest_dir", "staged_root", "checkpoint", "cond_image", "out")

    # ---- data: fetch actions/caption/GT from the manifest sample ----
    ds = build_rollout_dataset(args)
    print(f"[inpaint] dataset size={len(ds)}; using index {args.index}")
    sample = ds._load_one(args.index)
    video_cthw = sample["video"]  # [C,T,H,W] uint8
    C, T, H, W = video_cthw.shape
    print(f"[inpaint] source video {tuple(video_cthw.shape)} cap='{sample['ai_caption'][:60]}...'")

    # ---- replace frame 0 with the edited image ----
    img = Image.open(args.cond_image).convert("RGB")
    if img.size != (W, H):
        print(f"[inpaint] resizing cond image {img.size} -> ({W},{H})")
        img = img.resize((W, H), Image.LANCZOS)
    cond = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [C,H,W] uint8
    video_cthw = video_cthw.clone()
    video_cthw[:, 0] = cond
    sample["video"] = video_cthw
    print(f"[inpaint] frame 0 replaced with {args.cond_image}")

    # ---- model ----
    device = torch.device("cuda")
    model = load_world_model(args)
    model.to(device)

    batch, video_pm1 = build_rollout_batch(model, sample, device)

    # ---- generate ----
    from cosmos_framework.tools.visualize.video import save_img_or_video

    torch.manual_seed(args.seed)
    print(f"[inpaint] sampling {args.num_steps} steps, guidance={args.guidance}")
    with torch.no_grad():
        outputs = model.generate_samples_from_batch(
            batch, seed=[args.seed], num_steps=args.num_steps, guidance=args.guidance,
        )
        gen = model.decode(outputs["vision"][0])  # [-1,1], [1,3,T,H,W]
    gen01 = pm1_to_unit(gen)
    print(f"[inpaint] generated {tuple(gen01.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fps = int(sample["conditioning_fps"].item())
    save_img_or_video(gen01.squeeze(0), args.out.replace(".mp4", ""), fps=fps)
    # save GT (with inpainted first frame) for comparison
    gt_video = pm1_to_unit(video_pm1)
    save_img_or_video(gt_video.squeeze(0), args.out.replace(".mp4", "_gt"), fps=fps)
    print(f"[inpaint] wrote {args.out} and GT counterpart")


if __name__ == "__main__":
    main()
