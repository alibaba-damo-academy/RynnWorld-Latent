#!/usr/bin/env python3
"""Roll out a trained RynnWorld-Latent checkpoint on held-out chunks and compare to GT.

It does NOT use the stock ``cosmos_framework.scripts.inference`` CLI, because
that CLI's action path is incompatible with our data contract in two ways:

  * ``inference/action.py`` hardcodes ``raw_action_dim = EMBODIMENT_TO_RAW_ACTION_DIM
    [domain_name]`` (hand_pose -> 57), but our action is 608-dim.
  * it reads only ``action_chunk_size + 1`` frames from ``vision_path`` (21),
    whereas we condition on 81-pixel-frame chunks.

Instead we reuse ``RynnWorldManifestDataset`` to build a training-identical
sample (same 480p decode, same quantile normalization from action_stats.json,
same 20x608 action selection), then hand-build the forward_dynamics inference
batch with the SAME sequence plan the trainer computes
(``build_sequence_plan_from_mode(video_length=81, action_length=20)`` -> Case C,
offset 1), and drive ``generate_samples_from_batch`` + ``decode`` directly.

Usage::

    cd RynnWorld-Latent
    export PYTHONPATH=$PWD:$PWD/third_party/cosmos-framework
    python scripts/inference/rollout.py \
        --manifest-dir /path/to/manifest \
        --staged-root  /path/to/staged480 \
        --checkpoint   <IMAGINAIRE_OUTPUT_ROOT>/.../checkpoints/iter_000001000 \
        --out /tmp/rollout --num-samples 4 --num-steps 35 --guidance 1.5

Pass ``--dry-run`` to validate only the data-prep half (no model, no GPU); this
is testable before any checkpoint exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# cosmos's make_config() imports shipped action-policy experiments whose dataset
# wrappers import lerobot at module level; RynnWorld-Latent never uses lerobot datasets,
# so install the same inert stub scripts/train.py uses (single implementation in
# rynnworld_latent.lerobot_stub).
from rynnworld_latent.lerobot_stub import ensure as _ensure_lerobot_stub

_ensure_lerobot_stub()

from rynnworld_latent.inference import (
    build_rollout_batch,
    build_rollout_dataset,
    crop_common,
    load_world_model,
    pm1_to_unit,
    psnr,
    temporal_diff,
)


def main():
    ap = argparse.ArgumentParser(description="Roll out a RynnWorld-Latent checkpoint vs GT")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--staged-root", default="")
    ap.add_argument("--checkpoint", default="", help="DCP checkpoint dir (iter_XXXXXXXXX)")
    ap.add_argument("--experiment", default="rynnworld_latent_edge_manifest_fullft")
    ap.add_argument("--out", default="outputs/rollout")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--indices", default="", help="comma-separated dataset indices; default spreads evenly")
    ap.add_argument("--no-ema", action="store_true", help="load regular weights, not EMA")
    ap.add_argument("--dry-run", action="store_true", help="validate data-prep only; no model/GPU")
    args = ap.parse_args()

    ds = build_rollout_dataset(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(ds)
    if args.indices:
        idxs = [int(x) for x in args.indices.split(",")]
    else:
        step = max(1, n // max(1, args.num_samples))
        idxs = [(i * step) % n for i in range(args.num_samples)]
    print(f"[eval] dataset size={n}; evaluating indices {idxs}")

    if args.dry_run:
        from cosmos_framework.data.generator.action.transforms import build_sequence_plan_from_mode

        for i in idxs:
            s = ds._load_one(i)
            v, a = s["video"], s["action"]
            sp = build_sequence_plan_from_mode(
                mode="forward_dynamics", video_length=v.shape[1],
                action_length=a.shape[0], video_temporal_downsample=4,
            )
            print(
                f"  idx={i}: video={tuple(v.shape)} uint8 "
                f"action={tuple(a.shape)} range[{a.min():.3f},{a.max():.3f}] "
                f"domain={int(s['domain_id'])} fps={int(s['conditioning_fps'])}\n"
                f"          cond_vision={sp.condition_frame_indexes_vision} "
                f"n_cond_action={len(sp.condition_frame_indexes_action)} "
                f"offset={sp.action_start_frame_offset} cap={s['ai_caption'][:50]!r}"
            )
        print("[eval] dry-run OK (data-prep validated; no model loaded)")
        return

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --dry-run")
    device = "cuda"
    from cosmos_framework.tools.visualize.video import save_img_or_video

    model = load_world_model(args)
    for i in idxs:
        s = ds._load_one(i)
        batch, gt_pm1 = build_rollout_batch(model, s, device)
        with torch.no_grad():
            outputs = model.generate_samples_from_batch(
                batch, seed=[args.seed], num_steps=args.num_steps, guidance=args.guidance,
            )
            gen = model.decode(outputs["vision"][0])  # [-1,1], [1,3,T,H,W] (or [3,T,H,W])
        gen01 = pm1_to_unit(gen)
        gt01 = pm1_to_unit(gt_pm1)
        # gt_pm1 is already the unpadded content crop, so this only absorbs any
        # residual sub-16px size difference against the decode output.
        g, t = crop_common(gen01, gt01)
        # Static baseline: GT frame 0 held for the whole clip. A useful floor — a model
        # that ignores the actions and just freezes the conditioning frame scores this.
        static = t[:, :, :1].expand_as(t)
        print(
            f"  idx={i}: PSNR(gen,gt)={psnr(g, t):.2f} dB | "
            f"static-first-frame baseline={psnr(static, t):.2f} dB | "
            f"motion(gen)/motion(gt)={temporal_diff(g) / max(temporal_diff(t), 1e-8):.2f} | "
            f"shape={tuple(g.shape)}"
        )
        save_img_or_video(gen01.squeeze(0), str(out / f"gen_{i:07d}"), fps=int(s["conditioning_fps"]))
        save_img_or_video(gt01.squeeze(0), str(out / f"gt_{i:07d}"), fps=int(s["conditioning_fps"]))
    print(f"[eval] wrote {len(idxs)} rollouts to {out}")


if __name__ == "__main__":
    main()
